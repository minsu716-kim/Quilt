import torch.nn as nn
import torch.nn.functional as F
import torch

class TwoLayerNet(nn.Module):
    """TwoLayerNet.
    
    Attributes:
        input_dim: Number of input features.
        num_classes: Number of classes.
        hidden_units: Number of nodes in hidden layer.
        seed: Random seed number.
    """
    def __init__(self, input_dim, num_classes, hidden_units, seed):
        """Initialize TwoLayerNet."""
        super(TwoLayerNet, self).__init__()
        torch.manual_seed(seed)
        self.linear1 = nn.Linear(input_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, num_classes)
        self.feature_dim = hidden_units
    
    def forward(self, x, last=False, freeze=False):
        """Forward pass of the model.
        
        Args:
            x: Input data.
            last: If True, return outputs of hidden layer and last layer. 
                  If False, only return outputs of last layer.
            freeze: If True, we freeze model parameters. 
                    If False, we do not freeze model parameters.
        Returns:
            Model outputs.
        """
        if freeze:
            with torch.no_grad():
                l1scores = F.relu(self.linear1(x))
        else:
            l1scores = F.relu(self.linear1(x))
        scores = self.linear2(l1scores)
        if last:
            return scores, l1scores
        else:
            return scores

    def get_feature_dim(self):
        return self.feature_dim

    def get_embedding_dim(self):
        return self.feature_dim