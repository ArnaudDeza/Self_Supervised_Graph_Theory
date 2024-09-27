# Libraries
import copy
import random
import pickle
import torch.optim
import numpy as np
import time

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from ray.experimental.tqdm_ray import tqdm
from typing import Tuple, List, Optional, Union

# Import from other files
from core.gumbeldore_dataset import GumbeldoreDataset
from core.train import main_train_cycle

from graph_conjectures.config import GraphConfig
from graph_conjectures.trajectory import Trajectory as GraphTrajectory
from graph_conjectures.network import Graph_Theory_Network 


"""
Search for counterexamples of Graph Theory Conjectures using self-supervised learning.
===========================
"""


def get_network(config: GraphConfig, device: torch.device) -> Graph_Theory_Network:
    if config.architecture == "BQ":
        network = Graph_Theory_Network(config, device)
    else:
        raise ValueError(f"Unknown architecture {config.architecture}")
    return network







def generate_instances(config: GraphConfig):
    """
    Generate random instances for which we sample solutions to use as supervised signal.
    """
    if config.gumbeldore_config["active_search"] is None:
        num_instances = config.gumbeldore_config["num_instances_to_generate"]
        nodes = np.random.random((num_instances, config.num_nodes, 2))
        problem_instances = [{"inst": nodes[i]} for i in range(num_instances)]
    else:
        print(f"Active search with instances from {config.gumbeldore_config['active_search']}")
        with open(config.gumbeldore_config["active_search"], "rb") as f:
            problem_instances = pickle.load(f)

    return problem_instances, config.gumbeldore_config["batch_size_per_worker"], config.gumbeldore_config["batch_size_per_cpu_worker"]



def beam_leaves_to_result(trajectories: List[GraphTrajectory]):
    best_trajectory = sorted(trajectories, key=lambda y: y.objective)[0]
    return best_trajectory.partial_tour.copy(), best_trajectory.objective



def save_search_results_to_dataset(destination_path: str, problem_instances, results, append_to_dataset):
    """
    Assumes all problem instances to be of the same (num_jobs, num_machines)-size.
    Returns the mean generated objective.
    """
    # Each result in `results` is a tuple (tour, objective) (see above)
    dataset = [
        {"inst": instance["inst"], "tour": results[i][0]} for i, instance in enumerate(problem_instances)
    ]

    if not append_to_dataset:
        with open(destination_path, "wb") as f:
            pickle.dump(dataset, f)
    else:
        with open(destination_path, "rb") as f:
            instances = pickle.load(f)

        instances.extend(dataset)
        with open(destination_path, "wb") as f:
            pickle.dump(instances, f)

    return np.array([x[1] for x in results]).mean()




# TRAINING

def get_gumbeldore_dataloader(config: GraphConfig, network_weights: dict, append_to_dataset: bool):
    gumbeldore_dataset = GumbeldoreDataset(
        config=config,
        trajectory_cls=GraphTrajectory,
        generate_instances_fn=generate_instances,
        get_network_fn=get_network,
        beam_leaves_to_result_fn=beam_leaves_to_result,
        process_search_results_fn=save_search_results_to_dataset
    )
    mean_generated_obj = gumbeldore_dataset.generate_dataset(network_weights, append_to_dataset)
    print(f"Mean obj of generated data: {mean_generated_obj}")
    print("Training with generated data.")

    torch.cuda.empty_cache()

    time.sleep(10)
    # Load dataset.
    dataset = RandomTSPDataset(model_type=config.architecture,
                               expert_pickle_file=config.gumbeldore_config["destination_path"],
                               batch_size=config.batch_size_training,
                               data_augmentation=config.data_augmentation,
                               data_augmentation_linear_scale=config.data_augmentation_linear_scale,
                               augment_direction=config.augment_direction,
                               custom_num_instances=None,
                               custom_num_batches=config.custom_num_batches)

    return (DataLoader(dataset, batch_size=1, shuffle=True,
                       num_workers=config.num_dataloader_workers, pin_memory=True,
                       persistent_workers=True),
           float(mean_generated_obj))




def train_with_dataloader(config: GraphConfig, dataloader: DataLoader, network: Graph_Theory_Network, optimizer: torch.optim.Optimizer):
    """
    Iterates over dataloader and trains given network with optimizer.
    """
    network.train()

    accumulated_loss = 0
    num_batches = len(dataloader)
    progress_bar = tqdm(range(num_batches))
    data_iter = iter(dataloader)
    for _ in progress_bar:
        data = next(data_iter)
        data["nodes"] = data["nodes"][0].to(network.device)
        data["next_node_idx"] = data["next_node_idx"][0].to(network.device)

        logits = network(data)
        criterion = CrossEntropyLoss(reduction='mean')
        loss = criterion(logits, data["next_node_idx"])


        # Optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.optimizer["gradient_clipping"] > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

        optimizer.step()

        batch_loss = loss.item()
        accumulated_loss += batch_loss

        progress_bar.set_postfix({"batch_loss": batch_loss})

        del data

    avg_loss = accumulated_loss / num_batches
    return avg_loss





def train_for_one_epoch_gumbeldore(
                                   config: GraphConfig,
                                   network: Graph_Theory_Network,
                                   network_weights: dict,
                                   optimizer: torch.optim.Optimizer,
                                   append_to_dataset: bool
                                   ) -> Tuple[float, dict]:
    """
    Trains the network for one epoch using Gumbeldore-style data generation.

    Parameters:
        config: [TSPConfig] TSP Config object.
        network: [TSPNetwork] TSP Network object.
        network_weights: [dict] Network weights to use for generating data.
        optimizer: [torch.optim.Optimizer] Optimizer object.
        append_to_dataset: [bool] If True, appends the generated data to the existing dataset.

    Returns:
        avg_loss: [float] Average batch loss.
        dict: [dict] Dictionary with loggable values.
    """

    dataloader, mean_generated_obj = get_gumbeldore_dataloader(config, network_weights, append_to_dataset)
    avg_loss = train_with_dataloader(config, dataloader, network, optimizer)
    return avg_loss, {"Avg generated obj": float(mean_generated_obj)}



if __name__ == '__main__':
    print(f">> Searching for Counter-Examples for Graph Theory conjectures <<")
    config = GraphConfig()



    main_train_cycle(
        config=config,
        get_network_fn=get_network,
        train_for_one_epoch_gumbeldore_fn=train_for_one_epoch_gumbeldore
    )


