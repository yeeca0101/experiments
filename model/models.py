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
from torchvision.models import resnet18,resnet50,resnet152, swin_b,swin_s,swin_t
from g_mlp_pytorch import gMLPVision

from util.utils import pair

class BasicMLP(nn.Module):
    '''
    max acc 97.71 | 97.62 in MNIST
    '''
    def __init__(self, input_size, hidden_size, num_classes, activation=None,**kwargs):
        super(BasicMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.activation = activation
 
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x
    
class ResNet(nn.Module):
    support_models = {
        'resnet18': resnet18,
        'resnet50': resnet50,
        'resnet152': resnet152
    }
    def __init__(self, name: str, 
                 input_size=None,
                 in_channel=None,
                 n_classes=None,
                 pre_trained=False,
                 activation=None,
                 num_layers=None,
                 **kwargs,
                 ):
        super(ResNet, self).__init__()  # Initialize the nn.Module base class
        self.name = name
        if num_layers:
            from torchvision.models import ResNet as BaseResNet
            from torchvision.models.resnet import BasicBlock
            self.model = BaseResNet(BasicBlock,num_layers)
        else:           
            self.model = ResNet.support_models[name](pretrained=pre_trained)
            self.model.fc = nn.Linear(512,n_classes,bias=False)

        self.model.conv1 = nn.Conv2d(in_channel,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if activation is not None:
            replace_activations(self.model,nn.ReLU,activation)
    
    def forward(self,x):
        return self.model(x)

class gMlpVision(nn.Module):
    def __init__(self,name: str=None, 
                 input_size=None,
                 in_channel=None,
                 n_classes=None,
                 activation=nn.GELU(),
                 **kwargs,) -> None:
        super().__init__()
        self.model=gMLPVision(image_size=input_size,
                         num_classes=n_classes,
                         channels=in_channel,
                         **kwargs
                         )
        self.name=name
        if activation is not None:
            replace_activations(self.model,nn.GELU,activation)

    def forward(self,x):
        return self.model(x)
    
class SwinTransformer(nn.Module):
    support_models={
        'swin_b':swin_b,
        'swin_s':swin_s,
        'swin_t':swin_t
    }
    model_kwargs_swin_b={
        'patch_size':[4, 4], 
        'embed_dim':128,    
        'depths':[2, 2, 18, 2],
        'num_heads':[4, 8, 16, 32], 
        'window_size':[7, 7],
        'stochastic_depth_prob':0.5
    }
    model_kwargs_swin_s={
        'patch_size':[4, 4], 
        'embed_dim':96,    
        'depths':[2, 2, 18, 2],
        'num_heads':[3, 6, 12, 24], 
        'window_size':[7, 7],
        'stochastic_depth_prob':0.3
    }
    model_kwargs_swin_t={
        'patch_size':[4, 4], 
        'embed_dim':96,    
        'depths':[2, 2, 6, 2],
        'num_heads':[3, 6, 12, 24], 
        'window_size':[7, 7],
        'stochastic_depth_prob':0.2
    }

    def __init__(self, name: str, 
                 in_channel=None,
                 n_classes=None,
                 pre_trained=False,
                 activation=None,
                 **kwargs,
                 ):
        super(SwinTransformer, self).__init__()  # Initialize the nn.Module base class
        self.name = name
        self.model = SwinTransformer.support_models[name](pretrained=pre_trained)
        self.kwargs = eval(f'SwinTransformer.model_kwargs_{name}')
        embed_dim=self.kwargs['embed_dim']

        self.model.features[0][0] = nn.Conv2d(in_channel,embed_dim, kernel_size=(4, 4), stride=(4, 4))
        
        head_in_ch = embed_dim * 2 ** (len(self.kwargs['depths']) - 1)
        self.model.head = nn.Linear(head_in_ch,n_classes,bias=False)

        if activation is not None:
            replace_activations(self.model,nn.GELU,activation)
    
    def forward(self,x):
        return self.model(x)

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

def replace_activations(model, old_activation, new_activation, **kwargs):
    for name, module in model.named_children():
        if isinstance(module, old_activation):
            # Instantiate the new activation with the provided kwargs
            setattr(model, name, new_activation.__class__(**kwargs))
        else:
            # Recursively call replace_activations for submodules
            replace_activations(module, old_activation, new_activation, **kwargs)

    

if __name__ == '__main__':
    kwargs = {
    'input_size' :784,
    'hidden_size': 128,
    'num_classes': 10,
    
    }
    models_class = get_models_addr()
    model = build_model(model_class=models_class['BasicMLP'],activation=nn.PReLU(),**kwargs)
    print(model)
    replace_activations(model,nn.PReLU,nn.GELU)
    print(model)
    print(getattr(model,'activation'))
