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
