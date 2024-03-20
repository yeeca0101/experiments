'''
Model class : fixed activation function ArgName = 'activation' 
'''
from typing import List,Any
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import torch.nn as nn

class BasicMLP(nn.Module):
    '''
    max acc 97.71 | 97.62 in MNIST
    '''
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super(BasicMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.activation = activation
 
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x
    

def get_models_addr(return_type='dict'):
    # 전역 네임스페이스에서 nn.Module을 상속받는 클래스 찾기
    module_subclasses = {name: cls for name, cls in globals().items()
                         if isinstance(cls, type) and issubclass(cls, nn.Module) and cls is not nn.Module}

    # 인스턴스 생성
    instances = {name: cls for name, cls in module_subclasses.items()}

    # 반환 타입에 따라 딕셔너리 혹은 리스트 반환
    if return_type == 'dict':
        return instances
    else:
        return list(instances.values())
    
def build_model(model_class,activation,**kwargs):
    '''Model Class에 activation arg 있는 경우'''
    return model_class(activation=activation,**kwargs)

def replace_activations(model, 
                        prev_activation:List[nn.Module,]|nn.Module=(nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU), 
                        new_activation:nn.Module|None=None):
    """
    Recursively replaces all activation functions with the new activation in the model.

    Args:
    model (nn.Module): PyTorch model.
    prev_activation : target activation class, not instance.
    new_activation (nn.Module): The new activation function to use.

    ex. replace_activations(model,nn.PReLU,nn.GELU())
    """
    if not isinstance(prev_activation,tuple):
        prev_activation = (prev_activation)

    for name, child in model.named_children():
        # If the child is an activation function (e.g., ReLU, Sigmoid, etc.),
        # replace it with the new activation function.
        # You might need to add more activation types here.
        if isinstance(child, prev_activation):
            setattr(model, name, new_activation)
        # If the child is not an activation, recurse on it if it has children of its own.
        elif list(child.children()):
            replace_activations(child, prev_activation, new_activation)

if __name__ == '__main__':
    kwargs = {
    'input_size' :784,
    'hidden_size': 128,
    'num_classes': 10,
    
    }
    models_class = get_models_addr()
    model = build_model(model_class=models_class['BasicMLP'],activation=nn.PReLU(),**kwargs)
    print(model)
    replace_activations(model,nn.PReLU,nn.GELU())
    print(model)
    print(getattr(model,'activation'))
