import numpy as np

# dim1 = np.array([1, 2, 3, 4])
# print(dim1)
# print(dim1.ndim)  # 배열의 차원수 출력
# print(dim1.shape)  # 배열이 몇행 몇열로 이루어져 있는지 출력
# print(dim1.size)  # 해당 배열에 데이터가 총 몇개 있는지 출력
#
# dim2 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
# print(dim2)
# print(dim2.ndim)  # 배열의 차원수 출력
# print(dim2.shape)  # 배열이 몇행 몇열로 이루어져 있는지 출력
# print(dim2.size)  # 해당 배열에 데이터가 총 몇개 있는지 출력


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 1 / (1 + e^(-x))


# def identity_function(x):
#     return x


# 신경망을 생성하는 함수
def init_network():
    network = {
        'w1': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]), 'b1': np.array([0.1, 0.2, 0.3]),
        'w2': np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]), 'b2': np.array([0.1, 0.2]),
        'w3': np.array([[0.1, 0.3], [0.2, 0.4]]), 'b3': np.array([0.1, 0.2])
    }
    return network


# 신경망 내에서 일어나는 연산을 해주는 함수
def forward(network, x):
    w1, w2, w3 = network['w1'], network['w2'], network['w3']    # 은닉층
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    neuron1 = np.dot(x, w1) + b1
    activation1 = sigmoid(neuron1)

    neuron2 = np.dot(activation1, w2) + b2
    activation2 = sigmoid(neuron2)

    neuron3 = np.dot(activation2, w3) + b3
    print_result = neuron3

    return print_result


network = init_network()
x = np.array([1.0, 0.5])    # 입력층
y = forward(network, x)
print(y)    # 출력층 출력
