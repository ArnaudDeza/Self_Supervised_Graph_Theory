import torch
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from typing import Optional, List, Union
from core.abstracts import BaseTrajectory
from graph_conjectures.network import Network




class Trajectory(BaseTrajectory):
    """
    Represents a partial sequence of edge flipping decisions used for beam search/rolling out policy

    The idea is that we get a graph with n nodes, we will loop through every potential edge of the complete graph and 
    decide whether to flip it or not. The trajectory will be a sequence of edge flipping decisions.

    Once the trajectory is finished, we can evaluate the objective of the trajectory, which is how close the graph is
    to being a counterexample of a conjecture (using a scoring function)
    """
    def __init__(self,
                 num_nodes: int,
                 num_edges: int,
                 debug: bool = False):
        self.debug = debug
        self.num_nodes: int = num_nodes # number of nodes in the graph
        self.num_edges: int = num_edges # number of edges in the complete graph

        ############################################################################################################
        #                                 >>>>>>>>>>> trajectory attributes <<<<<<<<<<<<
        ############################################################################################################

        # objective value of the trajectory (should be set when trajectory is finished)
        self.objective: Optional[float] = None  

        # adjacency matrix of the original input graph
        self.init_adjacency_matrix: Optional[torch.FloatTensor] = None

        # adjacency matrix of the current graph
        self.current_adjacency_matrix: Optional[torch.FloatTensor] = None

        # edge flipping decisions, 1 if we flip the edge, 0 if we don't; shape (num_edges,)
        self.edge_flips: Optional[torch.FloatTensor] = None

        # index of current edge being considered
        self.current_edge_idx: int = 0

        # number of edges flipped so far
        self.num_edges_flipped: int = 0

        # number of edges left to flip
        self.num_edges_left: int = num_edges

        # list of edge indices to consider for flipping (ordered)
        self.edge_indices: List[int] = list(range(num_edges))

        # Node features for ML network
        self.nodes_features: Optional[torch.FloatTensor] = None

        # Edge features for ML network
        self.edges_features: Optional[torch.FloatTensor] = None


    @staticmethod
    def init_from_adjacency_matrix(nodes_features: torch.FloatTensor,
                                   edges_features: torch.FloatTensor,
                                   adjacency_matrix: torch.FloatTensor):

        
        """
        Initialize a trajectory from an adjacency matrix
        """

        # initialize the trajectory
        traj = Trajectory()

        # set the number of nodes and edges
        traj.num_nodes = adjacency_matrix.shape[0]
        traj.num_edges = int(traj.num_nodes * (traj.num_nodes - 1) / 2)

        # set the initial adjacency matrix of the input data point (i.e a graph)
        traj.init_adjacency_matrix = adjacency_matrix
        # set the current adjacency matrix of the graph that will be updated as we flip edges
        traj.current_adjacency_matrix = adjacency_matrix
        # initialize the edge flipping decisions to all zeros (will be updated as we flip edges)
        traj.edge_flips = torch.zeros(traj.num_edges)
        # set the current edge index to 0
        traj.current_edge_idx = 0
        # set the number of edges flipped so far to 0
        traj.num_edges_flipped = 0
        # set the number of edges left to flip to the total number of edges
        traj.num_edges_left = traj.num_edges
        # set the edge indices to consider for flipping
        traj.edge_indices = list(range(traj.num_edges))

        # set the node features
        traj.nodes_features = nodes_features
        # set the edge features
        traj.edges_features = edges_features

        return traj
    

    @staticmethod
    def init_batch_from_instance_list(instances, network, device: torch.device):
        """
        Instances are a list of problem instances. We will return a list of `Trajectory` objects

        This function is used to initialize a batch of trajectories from a batch of problem instances. 

        Parameters:
            instances [List[Instance]]: List of problem instances where each instances consists of a dictionary with 3 things:
                                            - adjacency_matrix: the adjacency matrix of the graph
                                            - node_features: the features of the nodes in the graph
                                            - edge_features: the features of the edges in the graph
            network [torch.nn.Module]: Policy/encoding network which might be used to encode the instance
            device [torch.device]: CPU/GPU Device on which to store tensors for fast access.
        
        """

        # step 1: Prepare the inputs to the encoder network by making a pytorch geometric data object i.e a batch of graphs
        # The encoder will return an updated embedding for each node's features (and edge features) in the graph
        # TODO: 

        # step 2: Pass the inputs to the encoder network to get the updated node and edge embeddings
        # TODO:

        # step 3: Return the list of trajectories
        return [Trajectory.init_from_adjacency_matrix(
            nodes_features=instance["nodes_features"],
            edges_features=instance["edges_features"],
            adjacency_matrix=instance["adjacency_matrix"]
        ) for instance in instances]
    


        nodes_batch = torch.cat([torch.from_numpy(instance["inst"]).float().to(device)[None, :, :] for instance in instances], dim=0)  # (B, num_nodes, 2)
        dist_mats_batch = torch.cdist(nodes_batch, nodes_batch, p=2.)

        if isinstance(network, LEHDPolicyNetwork):
            nodes_batch = network.encode(nodes_batch).detach().cpu()
            
        return [Trajectory.init_from_nodes(nodes=nodes_batch[i], distance_matrix=dist_mats_batch[i],) for i, instance in enumerate(instances)]

        
        
    @staticmethod
    def log_probability_fn(trajectories: List['Trajectory'],
                          network: Network,
                           to_numpy: bool) -> Union[torch.Tensor, List[np.array]]:
        """
        Given a list of trajectories and a policy network, returns a list of numpy arrays of length num_actions OR a torch tensor of shape (num_trajectories, num_actions), where each numpy array/tensor is a log-probability distribution over the next actions.

        Parameters:
            trajectories [List[Trajectory]]
            network [torch.nn.Module]: Policy network
            to_numpy [bool]: If True, the returned list of

        Returns:
            If `to_numpy` is False, it should return a torch.FloatTensor of shape (num_trajectories, num_actions), otherwise a list of numpy arrays.
        """
        with torch.no_grad():
            batch = Trajectory.trajectories_to_batch(trajectories)
            if isinstance(network, LEHDPolicyNetwork):
                policy_logits = network.decode(batch["nodes"])
            else:
                policy_logits = network(batch)

            batch_log_probs = torch.log_softmax(policy_logits, dim=1)
        if not to_numpy:
            return batch_log_probs
        
        batch_log_probs = batch_log_probs.cpu().numpy()
        return [batch_log_probs[i] for i in range(len(trajectories))]
    

    def transition_fn(self, action: int):
        """
        Function that takes the action 0 or 1 (0: don't flip edge, 1: flip edge) and returns a new trajectory obtained by executing the action on the trajectory.

        Parameters:
            action [int]: action to take (0 or 1)
        
        Returns:
            Tuple[Trajectory, bool]: A new trajectory obtained by executing the action on the trajectory and a boolean variable indicating whether the new trajectory has reached the end of the episode or not.
        """

        # Update the trajectory based on the action
        new_traj = self.flip_current_edge(action)
        # Check if the new trajectory is finished
        is_leaf = new_traj.num_edges_left == 0
        return new_traj, is_leaf
    
    def flip_current_edge(self, action: int):
        """
        Flip the current edge based on the action (0: don't flip edge, 1: flip edge)

        Parameters:
            action [int]: action to take (0 or 1)
        
        Returns:
            Trajectory: A new trajectory obtained by executing the action on the current trajectory
        """
        # Need to update current_adjacency_matrix, edge_flips, current_edge_idx, num_edges_flipped, num_edges_left

        # get the edge index of the current edge being considered
        edge_idx = self.edge_indices[self.current_edge_idx]
        # get the edge index in the adjacency matrix
        i, j = self.get_edge_indices(edge_idx)
        # get the current adjacency matrix
        adj_matrix = self.current_adjacency_matrix
        # get the current edge flip decisions
        edge_flips = self.edge_flips

        # update the adjacency matrix based on the action
        if action == 1:
            adj_matrix[i, j] = 1 - adj_matrix[i, j]
            adj_matrix[j, i] = 1 - adj_matrix[j, i]
        else:
            # do nothing
            pass

        # update the edge flip decisions
        edge_flips[edge_idx] = action

        # update the current edge index being considered
        self.current_edge_idx += 1
        # update the number of edges flipped based on action
        self.num_edges_flipped += action
        # update the number of edges left to flip
        self.num_edges_left -= 1

        # create a new trajectory and populate it with the updated attributes
        traj = Trajectory()

        # attributes that never change throughout the trajectory i.e. they are the same as the original trajectory
        traj.num_nodes = self.num_nodes
        traj.num_edges = self.num_edges
        traj.init_adjacency_matrix = self.init_adjacency_matrix

        # attributes that change throughout the trajectory
        traj.current_adjacency_matrix = adj_matrix
        traj.edge_flips = edge_flips
        traj.current_edge_idx = self.current_edge_idx
        traj.num_edges_flipped = self.num_edges_flipped
        traj.num_edges_left = self.num_edges_left
        traj.edge_indices = self.edge_indices



        #TODO: update the node and edge features
        traj.nodes_features = self.nodes_features
        traj.edges_features = self.edges_features




        # update the objective value of the trajectory if the trajectory is finished
        if self.num_edges_left <= 0:
            # compute the objective value of the trajectory
            traj.objective = self.compute_objective(adj_matrix)
        else:
            traj.objective = None

            

        return traj
    


    def get_edge_indices(self, edge_idx: int):
        """
        Given an edge index, return the indices of the edge in the adjacency matrix

        Parameters:
            edge_idx [int]: edge index
        
        Returns:
            Tuple[int, int]: indices of the edge in the adjacency matrix
        """
        i = 0
        while edge_idx >= self.num_nodes - i - 1:
            edge_idx -= self.num_nodes - i - 1
            i += 1
        return i, i + edge_idx + 1
    

    
    def compute_objective(self, adj_matrix: torch.FloatTensor):
        """
        Compute the objective value of the trajectory

        Parameters:
            adj_matrix [torch.FloatTensor]: adjacency matrix of the graph
        
        Returns:
            float: the objective value of the trajectory
        """
        INF  = 100000
        # compute the objective value of the trajectory
        N = adj_matrix.shape[0]
        adj_matrix = adj_matrix.flatten()
        G = nx.Graph()
        G.add_nodes_from(list(range(N)))
        count = 0
        for i in range(N):
            for j in range(i+1,N):
                if adj_matrix[count] == 1:
                    G.add_edge(i,j)
                count += 1

        #G is assumed to be connected in the conjecture. If it isn't, return a very negative reward.
        if not (nx.is_connected(G)):
            return -INF
            
        #Calculate the eigenvalues of G
        evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
        evalsRealAbs = np.zeros_like(evals)
        for i in range(len(evals)):
            evalsRealAbs[i] = abs(evals[i])
        lambda1 = max(evalsRealAbs)

        #Calculate the matching number of G
        maxMatch = nx.max_weight_matching(G)
        mu = len(maxMatch)
            
        #Calculate the reward. Since we want to minimize lambda_1 + mu, we return the negative of this.
        #We add to this the conjectured best value. This way if the reward is positive we know we have a counterexample.
        myScore = math.sqrt(N-1) + 1 - lambda1 - mu
        if myScore > 0:
            #You have found a counterexample. Do something with it.
            print("\n\n\n\n \t\t >>>>>>>> Counterexample found for n = {} !".format(N))
            print("adj_matrix")
            print(adj_matrix)

            nx.draw_kamada_kawai(G)
            plt.show()
            exit()
            
        return myScore



    def to_max_evaluation_fn(self) -> float:
        return -1. * self.objective
    

    @staticmethod
    def trajectories_to_batch(trajectories):
        """
        Converts a list of trajectories to a batch of tensors (all store in a dictionary)
    
        """
        #TODO

        #return {
        #
        #    "nodes": torch.stack([
        #        torch.cat((traj.start_node, traj.remaining_nodes, traj.dest_node), dim=0)
        #        if traj.start_node is not None else traj.remaining_nodes for traj in trajectories], dim=0).float().to(trajectories[0].distance_matrix.device)
        #}
    

        
