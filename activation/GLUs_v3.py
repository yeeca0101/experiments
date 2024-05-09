import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.utils import exclude_from_activations
from activation.acts import *

class BaseGLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden_channels = 2 * out_channels  # 게이팅 메커니즘을 위해 출력 채널의 2배로 설정
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding, bias=bias,groups=in_channels)
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias,groups=out_channels)
        self.dropout = nn.Dropout2d(drop)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x1, x2 = torch.chunk(x, 2, dim=1)  
        x1 = self.act_layer(x1)
        x = x1 + x2
        x = self.dropout(x)
        x = self.conv2(x)
        return x



class SwiGLU(BaseGLU):
    def __init__(self, in_planes, planes,drop=0.,**kwargs):
        super().__init__(in_channels=in_planes, out_channels=planes,act_layer=Swish,drop=drop)

class GLU(BaseGLU):
    def __init__(self, in_planes, planes,drop=0.,**kwargs):
        super().__init__(in_channels=in_planes, out_channels=planes,act_layer=nn.Sigmoid,drop=drop)

class SwiTGLU(BaseGLU):
    '''if requires_grad == False it same as SiLTGLU'''
    def __init__(self, in_planes, planes,drop=0.,**kwargs):
        super().__init__(in_channels=in_planes, out_channels=planes,act_layer=SwishT,drop=drop)

class GelGLU(BaseGLU):
    def __init__(self, in_planes, planes,drop=0.,**kwargs):
        super().__init__(in_channels=in_planes, out_channels=planes,act_layer=nn.GELU,drop=drop)

class ReGLU(BaseGLU):
    def __init__(self, in_planes, planes,drop=0.,**kwargs):
        super().__init__(in_channels=in_planes, out_channels=planes,act_layer=nn.ReLU,drop=drop)

class MiGLU(BaseGLU):
    def __init__(self, in_planes, planes,drop=0.,**kwargs):
        super().__init__(in_channels=in_planes, out_channels=planes,act_layer=nn.Mish,drop=drop)


def get_GLUs(return_type='dict'):
    # 전역 네임스페이스에서 nn.Module을 상속받는 클래스 찾기
    module_subclasses = {
        name: cls for name, cls in globals().items()
        if isinstance(cls, type) and issubclass(cls,BaseGLU)  # 클래스 확인 및 nn.Module 상속 확인
        and cls is not nn.Module and cls is not BaseGLU  # nn.Module과 BaseGLU 자체는 제외
        and not getattr(cls, '_exclude_from_activations', False)  # 특정 속성으로 제외
    }

    # 인스턴스 생성
    instances = {name: cls for name, cls in module_subclasses.items()}  # 클래스를 인스턴스화

    # 반환 타입에 따라 딕셔너리 혹은 리스트 반환
    if return_type == 'dict':
        return instances
    else:
        return list(instances.values())

# test

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        swiglu_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = swiglu_layer(inplanes, planes,drop=drop)
        self.bn1 = norm_layer(planes)
        self.relu = act_layer()
        self.conv2 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    



    
if __name__ == '__main__':
    from util.utils import count_parameters

    acts = get_GLUs('list')
    print(acts)
    # TrainableSwiGLU를 swiglu_layer로 사용하는 BasicBlock 정의
    basic_block = BasicBlock(
        inplanes=64,
        planes=128,
        stride=2,
        downsample=nn.Sequential(
            nn.Conv2d(64, 128 * BasicBlock.expansion, 1, stride=2, bias=False),
            nn.BatchNorm2d(128 * BasicBlock.expansion)
        ),
        swiglu_layer=SwiGLU,
        act_layer=nn.ReLU,  # TrainableSwiGLU 내부에서 SiLU를 사용하더라도 여기서는 ReLU를 사용할 수 있습니다.
        drop=0.1
    )

    # 임의의 입력 텐서 생성
    input_tensor = torch.rand(1, 64, 56, 56)  # 배치 크기 1, 채널 64, 56x56 크기

    # BasicBlock을 통과시키기
    output = basic_block(input_tensor)
    print(output.shape)  # 출력 텐서의 차원을 확인

    
    # for a in acts:
    #     act=a(in_planes=64,planes=128)
    #     a = act(torch.randn((4,64,28,28)))
    #     print(a.shape)
    #     print(count_parameters(act))