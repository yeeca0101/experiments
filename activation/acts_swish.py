import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import torch
import torch.nn as nn

from util.utils import exclude_from_activations


class NamedModule(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
    @property
    def __name__(self):
        return self.__class__.__name__
    
# final methods
@exclude_from_activations
class Swish(NamedModule):
    '''
        x*sigmoid(b*x)  ,b = trainable parameter
    '''
    def __init__(self, beta_init=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=True)

    def forward(self,x):
        return x*torch.sigmoid(self.beta*x)
    
@exclude_from_activations
class SwishT(NamedModule):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  # Learnable parameter
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x) + self.alpha * torch.tanh(x)

class SwishT_A(NamedModule):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        # self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        # base not used beta : swish-1 + tanh
        self.alpha = alpha  

    def backward_(self,x):
        fx = self.forward(x)
        return torch.sigmoid(x)*(x+self.alpha+1-fx)

    def forward(self, x):
        # simplify by x*torch.sigmoid(x)+self.alpha*torch.tanh(x/2)
        return torch.sigmoid(x)*(x+2*self.alpha)-self.alpha


class SwishT_B(NamedModule):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha  

    def backward_(self,x):
        fx = self.forward(x)
        return torch.sigmoid(self.beta*x)*(self.beta*(x+self.alpha-fx)+1)

    def forward(self, x):
        return torch.sigmoid(self.beta*x)*(x+2*self.alpha)-self.alpha

class SwishT_C(NamedModule):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha  

    def backward_(self,x):
        fx = self.forward(x)
        return torch.sigmoid(self.beta*x)*(self.beta*x+self.alpha+1-self.beta*fx)

    def forward(self, x):
        return torch.sigmoid(self.beta*x)*(x+2*self.alpha/self.beta)-self.alpha/self.beta





def get_activations(return_type='dict'):
    # 전역 네임스페이스에서 NamedModule을 상속받는 클래스 찾기, 단 _exclude_from_activations 속성이 없는 클래스만
    module_subclasses = {name: cls for name, cls in globals().items()
                         if isinstance(cls, type) 
                         and (issubclass(cls, nn.Module) or issubclass(cls, NamedModule))
                         and cls is not NamedModule
                         and not getattr(cls, '_exclude_from_activations', False)}  # Check for the exclusion marker

    instances = {name: cls() for name, cls in module_subclasses.items()}

    if return_type == 'dict':
        return instances
    else:
        return list(instances.values())

if __name__ == '__main__':
    auto_instances_dict = get_activations('list')
    # auto_instances_dict = get_activations('dict')
    print(auto_instances_dict)
    for act in auto_instances_dict:
        print(type(act))
        x = torch.linspace(-3,3,50)
        print(x.chunk(2,-1)[0].shape)
        x = torch.randn((100,))
        with torch.no_grad():
            print(act(x).shape)