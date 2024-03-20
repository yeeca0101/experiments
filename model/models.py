import torch.nn as nn

class BasicMLP(nn.Module):
    '''
    max acc 97.71 | 97.62 in MNIST
    '''
    def __init__(self, input_size, hidden_size, num_classes, activation_function):
        super(BasicMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.activation_function = activation_function
 
    def forward(self, x):
        x = self.activation_function(self.layer1(x))
        x = self.activation_function(self.layer2(x))
        x = self.layer3(x)
        return x
    

def get_models(return_type='dict'):
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