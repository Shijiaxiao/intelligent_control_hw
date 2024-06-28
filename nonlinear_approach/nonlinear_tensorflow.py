import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 生成数据
def generate_data():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    y = 0.5 * (1 + np.cos(x))
    return x, y

x, y = generate_data()

# 构建模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(1)
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# 训练模型
model.fit(x, y, epochs=100, batch_size=32, verbose=1)

# 预测
y_pred = model.predict(x)

# 绘制结果
plt.plot(x, y, label='y = 0.5 * (1 + cos(x))')
plt.plot(x, y_pred, label='Neural Network Prediction - By Tensorflow')
plt.legend()
plt.show()