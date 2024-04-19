import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.utils import exclude_from_activations


class BaseGLU(nn.Module):
    def __init__(self, in_planes, planes, depth_wise=False):
        super(BaseGLU, self).__init__()
        self.adjust_channels = (in_planes % 2 != 0)
        if self.adjust_channels:
            in_planes += 1
        if depth_wise:
            self.conv_v = nn.Conv2d(in_planes // 2, in_planes // 2, kernel_size=1, groups=in_planes // 2, bias=False)
            self.conv_w = nn.Conv2d(in_planes // 2, in_planes // 2, kernel_size=1, groups=in_planes // 2, bias=False)
            self.conv_w2 = nn.Conv2d(in_planes // 2, in_planes, kernel_size=1, bias=False)
        else:
            self.conv_v = nn.Conv2d(in_planes // 2, planes, kernel_size=1, bias=False)
            self.conv_w = nn.Conv2d(in_planes // 2, planes, kernel_size=1, bias=False)
            self.conv_w2 = nn.Conv2d(planes, in_planes, kernel_size=1, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        if self.adjust_channels:
            # 채널별 평균 풀링을 사용하여 채널 추가
            cwa = x.mean([1], keepdim=True)  
            x = torch.cat([x, cwa], dim=1)  

        x, gate = x.chunk(2, dim=1)
        x = self.conv_v(x)
        gate = self.conv_w(gate)
        gate_activation = self.gate_activation(gate)
        x = x * gate_activation
        x = self.conv_w2(x)
        return x

    def gate_activation(self, gate):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SwiGLU(BaseGLU):
    '''if requires_grad == False, it same as SiLGLU'''
    def __init__(self, in_planes, planes, depth_wise=False,requires_grad=True):
        super(SwiGLU, self).__init__(in_planes, planes, depth_wise)
        self.beta = nn.Parameter(torch.ones(1),requires_grad=requires_grad)  

    def gate_activation(self, gate):
        return gate * torch.sigmoid(self.beta * gate)

class SwiTGLU(BaseGLU):
    '''if requires_grad == False it same as SiLTGLU'''
    def __init__(self, in_planes, planes, depth_wise=False,alpha=0.1,requires_grad=True):
        super(SwiTGLU, self).__init__(in_planes, planes, depth_wise)
        self.beta = nn.Parameter(torch.ones(1),requires_grad=requires_grad)  
        self.alpha = alpha

    def gate_activation(self, gate):
        return gate * torch.sigmoid(self.beta * gate) + self.alpha*torch.tanh(gate)

class GelGLU(BaseGLU):
    def gate_activation(self, gate):
        return F.gelu(gate)

class ReGLU(BaseGLU):
    def gate_activation(self, gate):
        return F.relu(gate)

class MiGLU(BaseGLU):
    def gate_activation(self, gate):
        return gate * torch.tanh(F.softplus(gate))





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
        act=a(in_planes=64,planes=8)
        a = act(torch.randn((4,64,28,28)))
        print(a.shape)
        print(count_parameters(act))