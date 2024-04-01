import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.utils import exclude_from_activations

# final methods
class sASN(nn.Module):
    '''scaled Swish using tanh'''
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super(sASN, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  # Learnable parameter
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x) + self.alpha * torch.tanh(x)

# will remove
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

# To be compared
class GELU(nn.GELU):
    def __init__(self, approximate: str = 'none') -> None:
        super().__init__(approximate)

class ELU(nn.ELU):
    '''
        x                   x>0       
        alpha*(exp(x)-1)    x<=0
    '''
    def __init__(self, alpha: float = 1, inplace: bool = False) -> None:
        super().__init__(alpha, inplace)

class SiLU(nn.SiLU):
    '''
        x*sigmoid(x) as same as Swish-1
        for reinforcement learning
    '''
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

class Swish(nn.Module):
    '''
        x*sigmoid(b*x)  ,b = trainable parameter
    '''
    def __init__(self, beta_init=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=True)

    def forward(self,x):
        return x*torch.sigmoid(self.beta*x)

class Mish(nn.Mish):
    '''
        x*Tanh(Softplus(x)) , Softplus = x*tanh(ln(1+exp(x))
    '''
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

class Softplus(nn.Softplus):
    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super().__init__(beta, threshold)

# https://arxiv.org/pdf/2301.05993.pdf
# @exclude_from_activations
class SoftModulusQ(nn.Module):
    def __init__(self):
        super(SoftModulusQ, self).__init__()
    
    def forward(self, x):
        return torch.where(torch.abs(x) <= 1, x**2 * (2 - torch.abs(x)), torch.abs(x))

# cvpr 2023
@exclude_from_activations
class IIEU(nn.Module):
    def __init__(self, in_features=1
                 ):
        super(IIEU, self).__init__()
        # Term-B의 Sigmoid 함수에 대한 파라미터
        self.beta = nn.Parameter(torch.zeros(1, in_features, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, in_features, 1, 1))
        
        # 조절 가능한 임계값 η
        self.eta = nn.Parameter(torch.tensor(0.05))
    
    def forward(self, x):
        # 특징-필터 유사도 계산 (Term-S)
        magnitude = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        term_s = x / magnitude
        
        # Term-B 계산
        avgpool = F.adaptive_avg_pool1d(x, x.size()[-1])
        term_b = torch.sigmoid(self.beta * avgpool + self.gamma)
        
        # Approximated Similarity와 Adjuster 적용
        approx_similarity = (term_s + term_b) / 2  # 단순화된 예
        adjusted_similarity = torch.where(approx_similarity > self.eta, 
                                           approx_similarity, 
                                           self.eta * torch.exp(approx_similarity - self.eta))
        
        # 최종 활성화 함수 적용
        activated = adjusted_similarity * x
        return activated
    

# test
class sSigmoid(nn.Module):
    def __init__(self, beta_init=1.0):
        super(sSigmoid, self).__init__()
        self.beta = beta_init
    def forward(self, x):
        return x * torch.sigmoid(self.beta * torch.tanh(x)) 

class SwishTb(nn.Module):
    def __init__(self, beta_init=1.0):
        super(SwishTb, self).__init__()
        self.beta = beta_init
    def forward(self, x):
        return x * torch.sigmoid(torch.tanh(self.beta*x)) 

class Twish(nn.Module):
    '''scaled Swish using tanh'''
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super(Twish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  # Learnable parameter
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        return torch.tanh(x)*torch.sigmoid(self.beta * x)
    
class SliuT(nn.Module):
    '''scaled Swish using tanh'''
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super(SliuT, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  # Learnable parameter
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        return x*torch.sigmoid(x)+self.alpha*torch.tanh(self.beta*x)

@exclude_from_activations
class BDU(nn.Module):
    def __init__(self, beta_init=1.0, alpha=0.1):
        super(BDU, self).__init__()
        self.beta = beta_init
        self.div_2pi = 1/torch.sqrt(torch.as_tensor([2*torch.pi]))
    def forward(self, x):
        return self.div_2pi * torch.exp(-1*torch.square(x)*0.5)
@exclude_from_activations
class GELUa(nn.Module):
    '''approximate GELU by https://arxiv.org/pdf/1606.08415.pdf'''
    def __init__(self,):
        super(GELUa, self).__init__()
        self.a = 1.702
    def forward(self, x):
        return x*torch.sigmoid(self.a*x)




# https://arxiv.org/pdf/2301.05993.pdf Vallés-Pérez et al. (2023)
# @exclude_from_activations
class Modulus(nn.Module):
    def __init__(self):
        super(Modulus, self).__init__()
    
    def forward(self, x):
        return torch.abs(x)

@exclude_from_activations
class BipolarSigmoid(nn.Module):
    def __init__(self):
        super(BipolarSigmoid, self).__init__()
    
    def forward(self, x):
        return 2 * (1 / (1 + torch.exp(-x))) - 1
    
@exclude_from_activations
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

@exclude_from_activations
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
