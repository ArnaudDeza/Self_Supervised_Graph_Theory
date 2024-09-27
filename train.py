import os
os.environ["RAY_DEDUP_LOGS"] = "0"
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import copy
import ray
import numpy as np
from core.abstracts import Config

from tsp.bq_network import dict_to_cpu
from logger import Logger

from typing import Callable, Optional, Tuple





def save_checkpoint(checkpoint: dict, filename: str, config: Config):
    os.makedirs(config.results_path, exist_ok=True)
    path = os.path.join(config.results_path, filename)
    torch.save(checkpoint, path)


def main_train_cycle(
    config: Config,
    get_network_fn: Callable[[Config, torch.device], torch.nn.Module],
    train_for_one_epoch_gumbeldore_fn: Optional[Callable[
        [Config, torch.nn.Module, dict, torch.optim.Optimizer, bool], Tuple[float, Optional[dict]]]
    ] = None
):
    """
    Training cycle for all methods, supervised or Gumbeldore.

    Parameters:

        config: [Config] JSSP/TSP/CVRP/Gomoku Config object.

        get_network_fn: Callable[[Config, torch.device], torch.nn.Module]
                Method which takes a config object and a device and returns a fresh instance of a policy neural network
                on CPU, but with a device attribute.

        train_for_one_epoch_gumbeldore_fn: Optional[Callable[
            [Config, dict, torch.optim.Optimizer, bool], Tuple[float, Optional[dict]]]]
            Only relevant for Gumbeldore training. Method which takes in config object, network to train, network weights to use for
            generating data, optimizer object and a boolean variable indicating whether the generated data should be
            appended to the existing dataset. It then generates data Gumbeldore-style, trains for one epoch and
            returns the average batch loss as well as an optional dictionary with loggable values.

    """
    num_gpus = len(set([d for d in config.devices_for_eval_workers if d != "cpu"]))
    ray.init(num_gpus=num_gpus, logging_level="info")
    print(ray.available_resources())

    logger = Logger(config.results_path, config.log_to_file, config.log_to_mlflow, config.mlflow_server_uri)
    logger.log_hyperparams(config)

    # Fix random number generator seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Setup the network
    network = get_network_fn(config, torch.device(config.training_device))

    # Load checkpoint if needed
    if config.load_checkpoint_from_path is not None:
        print(f"Loading checkpoint from path {config.load_checkpoint_from_path}")
        checkpoint = torch.load(config.load_checkpoint_from_path)
        print(f"{checkpoint['epochs_trained']} episodes have been trained in the loaded checkpoint.")
        if config.reset_best_validation:
            print("Resetting best validation metric.")
            checkpoint["best_model_weights"] = None
            checkpoint["validation_opt_gap"] = float("inf")
    else:
        checkpoint = {
            "model_weights": None,
            "best_model_weights": None,
            "optimizer_state": None,
            "epochs_trained": 0,
            "validation_opt_gap": float("inf")
        }
    if checkpoint["model_weights"] is not None:
        network.load_state_dict(checkpoint["model_weights"])


    # backwards compatibility to have additional slot for best weights so far
    if "best_model_weights" not in checkpoint:
        checkpoint["best_model_weights"] = copy.deepcopy(checkpoint["model_weights"])\
            if checkpoint["model_weights"] is not None else None
    is_fresh_best_model = checkpoint["best_model_weights"] is None  # not needed for supervised

    print(f"Policy network is on device {config.training_device}")
    network.to(network.device)
    network.eval()


    if config.num_epochs > 0:

        # Training loop
        print(f"Starting training for {config.num_epochs} epochs.")

        best_model_weights = checkpoint["best_model_weights"]  # can be None
        best_validation_opt_gap = checkpoint["validation_opt_gap"]

        print("Setting up optimizer.")
        optimizer = torch.optim.Adam(network.parameters(),lr=config.optimizer["lr"],weight_decay=config.optimizer["weight_decay"])
        
        if checkpoint["optimizer_state"] is not None and config.load_optimizer_state:
            print("Loading optimizer state from checkpoint.")
            optimizer.load_state_dict(checkpoint["optimizer_state"] )


        print("Setting up LR scheduler")
        _lambda = lambda epoch: config.optimizer["schedule"]["decay_factor"] ** (checkpoint["epochs_trained"] // config.optimizer["schedule"]["decay_lr_every_epochs"])
        scheduler = LambdaLR(optimizer, lr_lambda=_lambda)

        for epoch in range(config.num_epochs):

            generated_loggable_dict = None
            
            # Decide whether the to be generated dataset should be appended to an existing
            # or if we should refresh it because we have a new best model
            append_to_current_dataset = config.gumbeldore_config["append_if_not_new_best"] and \
                                        not is_fresh_best_model and \
                                        not config.gumbeldore_config["active_search"] and \
                                        os.path.isfile(config.gumbeldore_config["destination_path"])
            print(f"Generating dataset. {'Appending to existing' if append_to_current_dataset else ''}")
            if best_model_weights is not None and config.gumbeldore_config["use_best_model_for_generation"]:
                print("Using weights of best model so far.")
                network_weights = copy.deepcopy(best_model_weights)
            else:
                network_weights = copy.deepcopy(network.get_weights())


            avg_loss, generated_loggable_dict = train_for_one_epoch_gumbeldore_fn(
                config, network, network_weights, optimizer, append_to_current_dataset
            )

            checkpoint["epochs_trained"] += 1
            scheduler.step()
            print(f">> Epoch {checkpoint['epochs_trained']}. Avg loss: {avg_loss}")
            logger.log_metrics({"Train avg loss": avg_loss}, step=epoch)
            
            if generated_loggable_dict is not None:
                logger.log_metrics(generated_loggable_dict, step=epoch)









            '''print("Validating...")
            torch.cuda.empty_cache()
            with torch.no_grad():
                validation_metric, validation_loggable_dict = validation_fn(config, network)
            print("Validation done.")
            print(validation_loggable_dict)
            logger.log_metrics(validation_loggable_dict, step=epoch)

            # Save model
            checkpoint["model_weights"] = copy.deepcopy(network.get_weights())
            checkpoint["optimizer_state"] = copy.deepcopy(
                dict_to_cpu(optimizer.state_dict())
            )
            checkpoint["validation_opt_gap"] = validation_metric

            save_checkpoint(checkpoint, "last_model.pt", config)
            is_fresh_best_model = False
            if validation_metric < best_validation_opt_gap:
                print(">> Got new best model.")
                is_fresh_best_model = True
                checkpoint["best_model_weights"] = copy.deepcopy(checkpoint["model_weights"])
                best_model_weights = checkpoint["best_model_weights"]
                best_validation_opt_gap = validation_metric
                save_checkpoint(checkpoint, "best_model.pt", config)'''

    
    
    print("\n\n \t >>>>>> Training done.")
    print("Finished. Shutting down ray.")
    ray.shutdown()

