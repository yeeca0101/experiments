from typing import List,Any
import torch.nn as nn
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

def vis_activations_with_derivates(acts, x, cols, xlim=None, ylim=None):
    """
    Visualizes the activations and their derivatives for a set of input data.
    
    Parameters:
    - acts: A dictionary with keys as the names of activation functions and values as instances of those functions.
    - x: The input tensor with `requires_grad=True`.
    - cols: The number of graphs to display per row.
    
    Example usage:
        acts = {
            'ReLU': nn.ReLU(),
            'SCiU_square': SCiU(pos_multiplier=1, neg_multiplier=1, clip_max=1, clip_min=-1),
            'SCiU_squarev2': SCiU(pos_multiplier=2, neg_multiplier=2, clip_max=1, clip_min=-1)
        }
        x = torch.linspace(-5, 5, 100, requires_grad=True)  # Example input tensor
        vis_activations(acts, x, cols=2)
    """
    total = len(acts)
    rows = total // cols + (1 if total % cols else 0)
    
    plt.figure(figsize=(cols * 10, rows * 4))  # Adjusting size for better visibility
    
    for i, (name, activation_instance) in enumerate(acts.items(), start=1):
        y = activation_instance(x)
        y.backward(torch.ones_like(x), retain_graph=True)
        
        # Plotting activation
        plt.subplot(rows, 2 * cols, 2 * i - 1)
        plt.plot(x.detach().numpy(), y.detach().numpy())
        plt.title(f'{name} Activation')
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        
        # Plotting derivative
        plt.subplot(rows, 2 * cols, 2 * i)
        plt.plot(x.detach().numpy(), x.grad.detach().numpy())
        plt.title(f'{name} Derivative')
        x.grad.zero_()  # Clear gradients for the next calculation
        
        if xlim:
            plt.xlim(xlim)
        # No ylim for derivative to auto-adjust
        
    plt.tight_layout()
    plt.show()

# for visualization landscape 
class SimpleNN(nn.Module):
    def __init__(self, activation_func):
        super(SimpleNN, self).__init__()
        self.activation_func = activation_func
        # Define a dummy layer to simulate a neural network operation
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.activation_func(self.fc1(x))  # Apply a linear transformation
        x = self.activation_func(self.fc2(x))  # Apply a linear transformation
        x = self.activation_func(self.fc3(x))  # Apply a linear transformation
        x = self.activation_func(self.fc4(x))  # Apply a linear transformation
        return x

def plot_output_landscape(network, resolution=400):
    linspace = torch.linspace(0, 10, resolution)
    x, y = torch.meshgrid(linspace, linspace)
    grid = torch.stack((x.flatten(), y.flatten()), dim=1)

    with torch.no_grad():
        output = network(grid).reshape(resolution, resolution)
    return output

# 2D
def visualize_activations_landscape(acts, resolution=400, cols=3):
    rows = (len(acts) + cols - 1) // cols
    plt.figure(figsize=(cols * 5, rows * 5))
    
    for i, (name, activation_func) in enumerate(acts.items(), 1):
        plt.subplot(rows, cols, i)
        network = SimpleNN(activation_func)
        output = plot_output_landscape(network, resolution)
        plt.imshow(output, extent=(0, 10, 0, 10), origin='lower', cmap='coolwarm')
        plt.colorbar()
        plt.title(name)
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
    plt.tight_layout()
    plt.show()

def plot_3d_output_landscape(network, resolution=400, input_type='linspace', input_range=(0, 10)):
    # Generate input grid based on the specified input type
    if input_type == 'linspace':
        linspace = torch.linspace(*input_range, resolution)
    elif input_type == 'gaussian':
        linspace = torch.normal(mean=(input_range[0] + input_range[1]) / 2,
                                std=(input_range[1] - input_range[0]) / 6,
                                size=(resolution,))
    elif input_type == 'uniform':
        linspace = torch.rand(resolution) * (input_range[1] - input_range[0]) + input_range[0]
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    x, y = torch.meshgrid(linspace, linspace)
    grid = torch.stack((x.flatten(), y.flatten()), dim=1)

    with torch.no_grad():
        output = network(grid).reshape(resolution, resolution)

    return x.numpy(), y.numpy(), output.numpy()

def visualize_3d_activations_landscape(acts, resolution=100,input_type='linspace', z_angle=-230):
    '''
    Example usage
    visualize_3d_activations(acts, resolution=100,input_type='linspace',z_angle=-230)
    '''

    num_activations = len(acts)
    cols = 3  # You can adjust this as needed
    rows = (num_activations + cols - 1) // cols  # Calculate the necessary number of rows
    margin = 0.5

    fig = plt.figure(figsize=(cols * 6, rows * 6))  # Adjust the figure size as needed
    for i, (name, activation_func) in enumerate(acts.items(), start=1):
        ax = fig.add_subplot(rows, cols, i, projection='3d')
        network = SimpleNN(activation_func)
        x, y, z = plot_3d_output_landscape(network, resolution,input_type=input_type)

        # Plot a 3D surface
        ax.view_init(elev=30, azim=z_angle)
        ax.plot_surface(x, y, z, cmap='coolwarm')
        ax.set_title(name)
        ax.set_xlim([0-margin, 10+margin])
        ax.set_ylim([10+margin, 0-margin])
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('Activation Output')

        ax.dist = 10

    plt.tight_layout()
    plt.show()




### utils
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def exclude_from_activations(cls):
    """
    Decorator to mark classes to be excluded from activation functions.
    """
    cls._exclude_from_activations = True  # Set an attribute to mark the class
    return cls

    

### build
def pair(x):
    if not isinstance(x,(list,tuple)):
        return (x,x)
    else:
        return x
    

