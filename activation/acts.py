import torch
import torch.nn as nn

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
    
# My method
class BipolarClippedUnit(nn.Module):
    def __init__(self, pos_multiplier=2, neg_multiplier=-2, clip_min=-8, clip_max=8):
        super(BipolarClippedUnit, self).__init__()
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
    # 전역 네임스페이스에서 nn.Module을 상속받는 클래스 찾기
    module_subclasses = {name: cls for name, cls in globals().items()
                         if isinstance(cls, type) and issubclass(cls, nn.Module) and cls is not nn.Module}

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
