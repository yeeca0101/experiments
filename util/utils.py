from typing import List,Any
import torch.nn as nn
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import torch

### vis
def vis_activations(activations, x, cols,xlim=None,ylim=None):
    """
    Visualizes the activations for a set of input data.
    
    Parameters:
    - activations: A list of activation function objects.
    - x: The input tensor.
    - cols: The number of graphs to display per row.
    
    Example usage:
        activations = [torch.relu, torch.sigmoid, torch.tanh]  # Add more activations as needed
        x = torch.linspace(-5, 5, 100)  # Example input tensor
        vis_activations(activations, x, cols=3)
    """
    total = len(activations)
    rows = total // cols + (1 if total % cols else 0)
    
    plt.figure(figsize=(cols * 5, rows * 4))
    
    for i, activation in enumerate(activations, 1):
        y = activation(x)
        plt.subplot(rows, cols, i)
        plt.scatter(x.numpy(), y.detach().numpy(), alpha=0.6)  # Adding alpha for better visualization if points overlap
        plt.title(activation.__class__.__name__)
        
        if ylim:
            plt.ylim(ylim)
        else:
            y_min, y_max = y.min().item(), y.max().item()
            margin = max((y_max - y_min) * 0.1, 0.1)  # Ensuring a minimum margin
            if y_min == y_max:  # Handling the case where all y values are identical
                y_min, y_max = y_min - 0.5, y_max + 0.5  # Defaulting to an arbitrary range
            else:
                y_min, y_max = y_min - margin, y_max + margin
            plt.ylim([y_min, y_max])
        if xlim:
            plt.xlim(xlim)
        
    
    plt.tight_layout()
    plt.show()


### utils
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
    
    

### build
def pair(x):
    if not isinstance(x,(list,tuple)):
        return (x,x)
    else:
        return x
    

