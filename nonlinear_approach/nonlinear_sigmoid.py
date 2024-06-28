import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # 初始化权重和偏置
        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i], layers[i + 1]) * 0.1
            bias = np.random.randn(1, layers[i + 1]) * 0.1
            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.activations = [x]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.sigmoid(z)
            self.activations.append(activation)

        return self.activations[-1]

    def backward(self, y_true):
        # 计算输出层的误差
        error = self.activations[-1] - y_true
        delta = error * self.sigmoid_derivative(self.activations[-1])

        # 初始化权重和偏置的梯度
        self.d_weights = [None] * len(self.weights)
        self.d_biases = [None] * len(self.biases)

        # 计算输出层的权重和偏置梯度
        self.d_weights[-1] = np.dot(self.activations[-2].T, delta)
        self.d_biases[-1] = np.sum(delta, axis=0, keepdims=True)

        # 反向传播到隐藏层
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * self.sigmoid_derivative(self.activations[i + 1])
            self.d_weights[i] = np.dot(self.activations[i].T, delta)
            self.d_biases[i] = np.sum(delta, axis=0, keepdims=True)

    def update_parameters(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(y)
            self.update_parameters()

            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.activations[-1]))
                print(f'Epoch {epoch}, Loss: {loss:.6f}')

    def predict(self, X):
        return self.forward(X)

# 生成数据
def generate_data():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
    y = 0.5 * (1 + np.cos(x))
    return x, y

x, y = generate_data()

# 构建神经网络
layers = [1, 10, 10, 1]  # 输入层1个节点，两个隐藏层各10个节点，输出层1个节点
nn = NeuralNetwork(layers, learning_rate=0.01)

# 训练模型
nn.train(x, y, epochs=10000)

# 预测
y_pred = nn.predict(x)

# 绘制结果
plt.plot(x, y, label='y = 0.5 * (1 + cos(x))')
plt.plot(x, y_pred, label='Neural Network Prediction - By Shijiaxiao - Sigmoid')
plt.legend()
plt.show()