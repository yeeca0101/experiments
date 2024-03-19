import torch.nn as nn

class BasicMLP(nn.Module):
    '''
    max acc 97.71 | 97.62 in MNIST
    '''
    def __init__(self, input_size, hidden_size, num_classes, activation_function):
        super(BasicMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.activation_function = activation_function
 
    def forward(self, x):
        x = self.activation_function(self.layer1(x))
        x = self.activation_function(self.layer2(x))
        x = self.layer3(x)
        return x