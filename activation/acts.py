import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# from activation.GLUs_v2 import *
from util.utils import exclude_from_activations


class NamedModule(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
    @property
    def __name__(self):
        return self.__class__.__name__
    


# final methods
@exclude_from_activations
class SwishT(NamedModule):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  # Learnable parameter
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x) + self.alpha * torch.tanh(x)

# variants of SwishT

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
    

# for comparsion
# [ACON_C, Pserf, ErfAct, SMU, GELU, SiLU, Mish, Swish]
class SMU(NamedModule):
    '''
    Implementation of SMU activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha: hyper parameter
    References:
        - See related paper:
        https://arxiv.org/abs/2111.04682
    '''
    def __init__(self, alpha = 0.25, mu = 1.0):
        '''
        Initialization.
        INPUT:
            - alpha: hyper parameter
            aplha is initialized with zero value by default
        '''
        super().__init__()
        self.alpha = alpha
        # initialize mu
        self.mu = nn.Parameter(torch.tensor([mu]),requires_grad=True)
       
    def forward(self, x):
        return ((1+self.alpha)*x + (1-self.alpha)*x*torch.erf(self.mu*(1-self.alpha)*x))/2

@exclude_from_activations
class GELU(nn.GELU):
    def __init__(self, approximate: str = 'none') -> None:
        super().__init__(approximate)
    @property
    def __name__(self):
        return self.__class__.__name__

@exclude_from_activations    
class SiLU(nn.SiLU):
    '''
        x*sigmoid(x) as same as Swish-1
        for reinforcement learning
    '''
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

    @property
    def __name__(self):
        return self.__class__.__name__

@exclude_from_activations
class Mish(nn.Mish):
    '''
        x*Tanh(Softplus(x)) , Softplus = x*tanh(ln(1+exp(x))
    '''
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

    @property
    def __name__(self):
        return self.__class__.__name__
    
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
    

# Definition of the ErfAct activation function
@exclude_from_activations
class ErfActFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha, beta):
        ctx.save_for_backward(x, alpha, beta)
        return x * torch.erf(alpha * torch.exp(beta * x))
 
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, beta = ctx.saved_tensors
        grad_input = grad_alpha = grad_beta = None
       
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * (torch.erf(alpha * torch.exp(beta * x)) +
                                        2 * alpha * beta * x * torch.exp(beta * x - (alpha * torch.exp(beta * x)) ** 2) /
                                        torch.sqrt(torch.tensor(torch.pi)))
       
        if ctx.needs_input_grad[1]:
            grad_alpha = grad_output * x * torch.exp(beta * x) * \
                         (-2 * (alpha * torch.exp(beta * x)) ** 2) / \
                         torch.sqrt(torch.tensor(torch.pi))
 
        if ctx.needs_input_grad[2]:
            grad_beta = grad_output * x ** 2 * alpha * torch.exp(beta * x) * \
                        (-2 * (alpha * torch.exp(beta * x)) ** 2) / \
                        torch.sqrt(torch.tensor(torch.pi))
 
        return grad_input, grad_alpha, grad_beta

class ErfAct(NamedModule):
    def __init__(self, alpha=0.75, beta=0.75):
        super(ErfAct, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.float32))
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.float32))
 
    def forward(self, x):
        # return ErfActFunction.apply(x, self.alpha, self.beta)
        return x * torch.erf(self.alpha * torch.exp(self.beta * x))
    
   
# Definition of the Pserf activation function
@exclude_from_activations
class PserfFunction(Function):
    @staticmethod
    def forward(ctx, input, gamma, delta):
        ctx.save_for_backward(input, gamma, delta)
        return input * torch.erf(gamma * torch.log1p(torch.exp(delta * input)))
 
    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, delta = ctx.saved_tensors
        grad_input = grad_gamma = grad_delta = None
       
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * (torch.erf(gamma * torch.log1p(torch.exp(delta * input))) +
                                        2 * gamma * delta * input / torch.sqrt(torch.tensor(torch.pi)) /
                                        (1 + torch.exp(-delta * input)) *
                                        torch.exp(-((gamma * torch.log1p(torch.exp(delta * input))) ** 2)))
       
        if ctx.needs_input_grad[1]:
            grad_gamma = grad_output * input * torch.log1p(torch.exp(delta * input)) * \
                         2 / torch.sqrt(torch.tensor(torch.pi)) * \
                         torch.exp(-((gamma * torch.log1p(torch.exp(delta * input))) ** 2))
 
        if ctx.needs_input_grad[2]:
            grad_delta = grad_output * input ** 2 * gamma / \
                         (1 + torch.exp(-delta * input)) * \
                         2 / torch.sqrt(torch.tensor(torch.pi)) * \
                         torch.exp(-((gamma * torch.log1p(torch.exp(delta * input))) ** 2))
 
        return grad_input, grad_gamma, grad_delta

class Pserf(NamedModule):
    def __init__(self, gamma=1.25, delta=0.85):
        super(Pserf, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.delta = nn.Parameter(torch.tensor(delta))

    def forward(self, x):
        # return PserfFunction.apply(input, self.gamma, self.delta)
        return x * torch.erf(self.gamma * torch.log1p(torch.exp(self.delta * x)))
    
@exclude_from_activations
class ACON_C(NamedModule):
    def __init__(self, p1=1.0, p2=0.0, beta=1.0):
        super(ACON_C, self).__init__()
        self.p1 = nn.Parameter(torch.tensor([p1]))
        self.p2 = nn.Parameter(torch.tensor([p2]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * (self.p1 - self.p2) * x) + self.p2 * x


# for appendix
@exclude_from_activations
class SiLUT(SwishT):
    def __init__(self,):
        super().__init__(beta_init=1.0, alpha=0.1,requires_grad=False)

    @property
    def __name__(self):
        return self.__class__.__name__

@exclude_from_activations
class SliuT(nn.Module):
    '''scaled Swish using tanh'''
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super(SliuT, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  # Learnable parameter
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        return x*torch.sigmoid(x)+self.alpha*torch.tanh(self.beta*x)

@exclude_from_activations
class ASN(NamedModule):
    'Adaptive Squared Non-Linearity'
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

@exclude_from_activations
class ELU(nn.ELU):
    '''
        x                   x>0       
        alpha*(exp(x)-1)    x<=0
    '''
    def __init__(self, alpha: float = 1, inplace: bool = False) -> None:
        super().__init__(alpha, inplace)


@exclude_from_activations
class Softplus(nn.Softplus):
    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super().__init__(beta, threshold)

# https://arxiv.org/pdf/2301.05993.pdf
@exclude_from_activations
class SoftModulusQ(NamedModule):
    def __init__(self):
        super(SoftModulusQ, self).__init__()
    
    def forward(self, x):
        return torch.where(torch.abs(x) <= 1, x**2 * (2 - torch.abs(x)), torch.abs(x))

# cvpr 2023
@exclude_from_activations
class IIEU(NamedModule):
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



@exclude_from_activations
class BDU(NamedModule):
    def __init__(self, beta_init=1.0, alpha=0.1):
        super(BDU, self).__init__()
        self.beta = beta_init
        self.div_2pi = 1/torch.sqrt(torch.as_tensor([2*torch.pi]))
    def forward(self, x):
        return self.div_2pi * torch.exp(-1*torch.square(x)*0.5)
    

# https://arxiv.org/pdf/2301.05993.pdf Vallés-Pérez et al. (2023)
@exclude_from_activations
class Modulus(NamedModule):
    def __init__(self):
        super(Modulus, self).__init__()
    
    def forward(self, x):
        return torch.abs(x)

@exclude_from_activations
class BipolarSigmoid(NamedModule):
    def __init__(self):
        super(BipolarSigmoid, self).__init__()
    
    def forward(self, x):
        return 2 * (1 / (1 + torch.exp(-x))) - 1
    
@exclude_from_activations
class TanhExp(NamedModule):
    def __init__(self):
        super(TanhExp, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))
    

def get_activations(return_type='dict'):
    # 전역 네임스페이스에서 NamedModule을 상속받는 클래스 찾기, 단 _exclude_from_activations 속성이 없는 클래스만
    module_subclasses = {name: cls for name, cls in globals().items()
                         if isinstance(cls, type) 
                         and (issubclass(cls, nn.Module) or issubclass(cls, NamedModule))
                         and cls is not NamedModule
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
    auto_instances_dict = get_GLUs('list')
    # auto_instances_dict = get_activations('dict')
    print(auto_instances_dict)
    act = auto_instances_dict[0]
    print(type(act))
    x = torch.linspace(-3,3,50)
    print(x.chunk(2,-1)[0].shape)
    x = torch.randn((100,))
    with torch.no_grad():
        print(act(x).shape)
