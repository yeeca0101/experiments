import torch.nn as nn
import matplotlib.pyplot as plt

def vis_activations(activations, x, cols):
    """
    activations: 활성화 함수 객체의 리스트
    x: 입력 텐서
    cols: 하나의 행에 표시될 그래프의 수
    """
    # 전체 활성화 함수의 수
    total = len(activations)
    # 계산된 행 수
    rows = total // cols + (1 if total % cols else 0)
    
    plt.figure(figsize=(cols * 5, rows * 4))
    
    for i, activation in enumerate(activations, 1):
        y = activation(x)
        plt.subplot(rows, cols, i)
        plt.scatter(x.numpy(), y.detach().numpy())
        plt.title(activation.__class__.__name__)
    
    plt.tight_layout()
    plt.show()


def get_Module_Class(return_type='dict'):
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