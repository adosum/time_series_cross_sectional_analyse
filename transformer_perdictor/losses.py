
from torch import nn
import torch

def gaussian_nll_loss(mu, sigma, target):
    """Gaussian Negative Log-Likelihood"""
    var = sigma ** 2
    loss = 0.5 * torch.log(var + 1e-6) + 0.5 * ((target - mu) ** 2 / (var + 1e-6))
    return loss.mean()
class UncertaintyLoss(nn.Module):
    """
    Dynamically weights multi-task losses. 
    Tasks with higher variance (noise) get their loss down-weighted automatically.
    """
    def __init__(self, num_tasks=5):
        super().__init__()
        # Learnable log variances initialized to zero
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        Args:
            losses: A list or tuple of scalar tensors containing the loss for each task.
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            # Precision = exp(-log_var). 
            # We use log_var for numerical stability instead of learning variance directly.
            precision = torch.exp(-self.log_vars[i])
            
            # Loss formula: L_i * exp(-var) + 0.5 * var
            total_loss += precision * loss + 0.5 * self.log_vars[i]
            
        return total_loss