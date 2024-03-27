import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import torch
import torch.nn as nn

from util.utils import exclude_from_activations

# final methods
class sASN(nn.Module):
    def __init__(self, beta_init=1.0, alpha=0.1):
        super(sASN, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]))  # Learnable parameter
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x) + self.alpha * torch.tanh(x)
    
class GELU(nn.GELU):
    def __init__(self, approximate: str = 'none') -> None:
        super().__init__(approximate)

class ASN(nn.Module):
    def __init__(self, beta_init=1.0, alpha=0.1):
        super(ASN, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=True)  # Learnable parameter
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        sig_part = torch.sigmoid(self.beta * x)
        sqr_part = torch.pow(x, 2)
        y = x * sig_part + self.alpha * sqr_part
        y = torch.clamp(y, -5, 5)
        # x = x * torch.sigmoid(self.beta * x) + self.alpha * torch.pow(x, 2)
        # print(x)
        return y



# https://arxiv.org/pdf/2301.05993.pdf
class SoftModulusQ(nn.Module):
    def __init__(self):
        super(SoftModulusQ, self).__init__()
    
    def forward(self, x):
        return torch.where(torch.abs(x) <= 1, x**2 * (2 - torch.abs(x)), torch.abs(x))

# https://arxiv.org/pdf/2301.05993.pdf Vallés-Pérez et al. (2023)
class Modulus(nn.Module):
    def __init__(self):
        super(Modulus, self).__init__()
    
    def forward(self, x):
        return torch.abs(x)

#
class BipolarSigmoid(nn.Module):
    def __init__(self):
        super(BipolarSigmoid, self).__init__()
    
    def forward(self, x):
        return 2 * (1 / (1 + torch.exp(-x))) - 1
    
#
class TanhExp(nn.Module):
    def __init__(self):
        super(TanhExp, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))
    
# My method : SquaredClipUnit
@exclude_from_activations
class UnitT(nn.Module):
    def __init__(self, pos_multiplier=2, neg_multiplier=-2, clip_min=-8, clip_max=8,pos_method=lambda x:x,neg_method=lambda x:x):
        super(UnitT, self).__init__()
        self.pos_multiplier = pos_multiplier
        self.neg_multiplier = neg_multiplier
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.pos_m = pos_method
        self.neg_m = neg_method

    def forward(self, x):
        y = torch.where(x > 0, self.pos_multiplier * self.pos_m(x),
                        self.neg_multiplier * self.neg_m(x))
        if self.clip_min and self.clip_max:
            y_clipped = torch.clamp(y, self.clip_min, self.clip_max)
        else:
            y_clipped =y
        return y_clipped
    
@exclude_from_activations
class BLU(nn.Module):
    def __init__(self, pos_multiplier=2, neg_multiplier=-2, clip_min=-8, clip_max=8):
        super(BLU, self).__init__()
        self.pos_multiplier = pos_multiplier
        self.neg_multiplier = neg_multiplier
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x):
        # 조건에 따라 값을 계산
        y = torch.where(x > 0, self.pos_multiplier * torch.log(x),
                        self.neg_multiplier * torch.log(torch.abs(x)))
        # 결과값 클리핑
        y_clipped = torch.clamp(y, self.clip_min, self.clip_max)
        return y_clipped

class SCiU(nn.Module):
    def __init__(self, pos_multiplier=2, neg_multiplier=-2, clip_min=-8, clip_max=8):
        super(SCiU, self).__init__()
        self.pos_multiplier = pos_multiplier
        self.neg_multiplier = neg_multiplier
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x):
        # 조건에 따라 값을 계산
        y = torch.where(x > 0, self.pos_multiplier * torch.square(x),
                        self.neg_multiplier * torch.square(x))
        # 결과값 클리핑
        y_clipped = torch.clamp(y, self.clip_min, self.clip_max)
        return y_clipped

def get_activations(return_type='dict'):
    # 전역 네임스페이스에서 nn.Module을 상속받는 클래스 찾기, 단 _exclude_from_activations 속성이 없는 클래스만
    module_subclasses = {name: cls for name, cls in globals().items()
                         if isinstance(cls, type) 
                         and issubclass(cls, nn.Module) 
                         and cls is not nn.Module
                         and not getattr(cls, '_exclude_from_activations', False)}  # Check for the exclusion marker

    # 인스턴스 생성
    instances = {name: cls() for name, cls in module_subclasses.items()}

    # 반환 타입에 따라 딕셔너리 혹은 리스트 반환
    if return_type == 'dict':
        return instances
    else:
        return list(instances.values())

if __name__ == '__main__':
    # act = BipolarSigmoid()
    # x = torch.linspace(-3,3,50)
    # x = torch.randn((100,))
    # plt.scatter(x.numpy(),act(x).numpy())
    auto_instances_dict = get_activations('dict')
    print(auto_instances_dict)
    print(type(auto_instances_dict['Modulus']))
    act = auto_instances_dict['Modulus']
    x = torch.linspace(-3,3,50)
    x = torch.randn((100,))
    print(act(x).numpy())
