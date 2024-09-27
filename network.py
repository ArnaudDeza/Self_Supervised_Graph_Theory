import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear

from config import GraphConfig


class Graph_Theory_Network(nn.Module):
    """
    Policy network for creating counter-examples to Graph Theory conjectures
    """
    def __init__(self, config: GraphConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device

        # TODO: Implement the network architecture

    def encode(self, x):
        #TODO: 
        pass
    
    def decode(self, x):
        #TODO
        pass
        
    def forward(self, x:dict):
        """
        Full pass through encoder and decoder as used in training.
        """
        #TODO
        pass


