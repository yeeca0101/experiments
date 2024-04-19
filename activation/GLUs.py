import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.utils import exclude_from_activations


class SwiTGLU(nn.Module):
    def __init__(self,beta_init=1.0,alpha=0.1,requires_grad=True,in_planes=None,planes=None) -> None:
        super().__init__()
        # if beta = 1, requires_grad=False, swish-1=silu
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha
        self.w = nn.Conv2d(in_planes//2, planes, kernel_size=1, bias=False)
        self.v = nn.Conv2d(in_planes//2, planes, kernel_size=1, bias=False)
        self.w2 = nn.Conv2d(planes, in_planes, kernel_size=1, bias=False)
        
    def forward(self,x:torch.Tensor):
        x, gate = x.chunk(2,dim=1)
        x = self.v(x)
        gate = self.w(gate)
        x = x*(gate*torch.sigmoid(self.beta*gate)+self.alpha*torch.tanh(gate))
        return self.w2(x)

class SwiGLU(nn.Module):
    def __init__(self,beta_init=1.0,requires_grad=True,in_planes=None,planes=None)  -> None:
        super().__init__()
        # if beta = 1, requires_grad=False, swish-1=silu
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.w = nn.Conv2d(in_planes//2, planes, kernel_size=1, bias=False)
        self.v = nn.Conv2d(in_planes//2, planes, kernel_size=1, bias=False)
        self.w2 = nn.Conv2d(planes, in_planes, kernel_size=1, bias=False)

    def forward(self,x:torch.Tensor):
        x, gate = x.chunk(2,dim=1)
        x = self.v(x)
        gate = self.w(gate)
        x = x*gate*torch.sigmoid(self.beta*gate)
        return self.w2(x)

class GLU(nn.Module):
    def __init__(self,in_planes=None,planes=None)  -> None:
        super().__init__()
        # if beta = 1, requires_grad=False, swish-1=silu
        self.w = nn.Conv2d(in_planes//2, planes, kernel_size=1, bias=False)
        self.v = nn.Conv2d(in_planes//2, planes, kernel_size=1, bias=False)
        self.w2 = nn.Conv2d(planes, in_planes, kernel_size=1, bias=False)

    def forward(self,x:torch.Tensor):
        x, gate = x.chunk(2,dim=1)
        x = self.v(x)
        gate = self.w(gate)
        x = x*torch.sigmoid(gate)
        return self.w2(x)

# test
class SwiTGLUv2(nn.Module):
    def __init__(self,beta_init=1.0,alpha=0.1,requires_grad=True,in_planes=None,planes=None) -> None:
        super().__init__()
        # if beta = 1, requires_grad=False, swish-1=silu
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha
        self.w = nn.Conv2d(in_planes//2, planes, kernel_size=1, bias=False)
        self.v = nn.Conv2d(in_planes//2, planes, kernel_size=1, bias=False)
        self.w2 = nn.Conv2d(planes, in_planes, kernel_size=1, bias=False)
        
    def forward(self,x:torch.Tensor):
        x, gate = x.chunk(2,dim=1)
        x = self.v(x)
        gate = self.w(gate)
        x = x*(gate*torch.sigmoid(self.beta*gate))+self.alpha*torch.tanh(gate)
        return self.w2(x)




def get_GLUs(return_type='dict'):
    # 전역 네임스페이스에서 nn.Module을 상속받는 클래스 찾기, 단 _exclude_from_activations 속성이 없는 클래스만
    module_subclasses = {name: cls for name, cls in globals().items()
                         if isinstance(cls, type) 
                         and issubclass(cls, nn.Module) 
                         and cls is not nn.Module
                         and not getattr(cls, '_exclude_from_activations', False)}  # Check for the exclusion marker

    # 인스턴스 생성
    instances = {name: cls for name, cls in module_subclasses.items()}

    # 반환 타입에 따라 딕셔너리 혹은 리스트 반환
    if return_type == 'dict':
        return instances
    else:
        return list(instances.values())
    
if __name__ == '__main__':
    from util.utils import count_parameters

    acts = get_GLUs('list')
    for a in acts:
        act=a(in_planes=22,planes=6)
        a = act(torch.randn((4,22,22,22)))
        print(a.shape)
        print(count_parameters(act))