import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 定义RNN类
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重和偏差
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.inputs, self.hs, self.ys = {}, {}, {}

        for t, x in enumerate(inputs):
            self.inputs[t] = x
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            self.hs[t] = h
            y = np.dot(self.Why, h) + self.by
            self.ys[t] = y

        return self.ys

# 定义正弦函数和导数余弦函数
def sine_function(x):
    return np.sin(x)

def cosine_derivative_function(x):
    return -np.cos(x)

# 生成训练数据
num_points = 100
time_steps = 10

# 生成正弦函数和导数余弦函数的数据
x_values = np.linspace(0, 2 * np.pi, num_points)
sine_values = sine_function(x_values)
derivative_values = cosine_derivative_function(x_values)

# 准备训练数据
input_data = [sine_values[i:i+time_steps] for i in range(num_points-time_steps)]
target_data = [derivative_values[i:i+time_steps] for i in range(num_points-time_steps)]

input_data = np.array(input_data).reshape(num_points-time_steps, time_steps, 1)
target_data = np.array(target_data).reshape(num_points-time_steps, time_steps, 1)

# 定义模型参数
input_size = 1
hidden_size = 32
output_size = 1

# 创建RNN模型
rnn = SimpleRNN(input_size, hidden_size, output_size)

# 定义训练参数
learning_rate = 0.01
epochs = 1000

# 训练模型
# ...

# 训练模型
for epoch in range(epochs):
    total_loss = 0

    for i in range(len(input_data)):
        inputs = input_data[i]
        targets = target_data[i]

        # 前向传播
        predictions = rnn.forward(inputs)

        # 计算损失
        loss = 0
        for t in range(time_steps):
            loss += np.mean((predictions[t] - targets[t]) ** 2)
        total_loss += loss / time_steps

        # 反向传播
        dL_dWhy = np.dot((predictions[time_steps-1] - targets[time_steps-1]).reshape(output_size, 1), rnn.hs[time_steps-1].T)
        dL_dby = np.sum(predictions[time_steps-1] - targets[time_steps-1])

        dL_dWxh, dL_dWhh, dL_dbh = 0, 0, 0
        dh_next = np.zeros_like(rnn.hs[0])

        for t in reversed(range(time_steps)):
            dh = np.dot(rnn.Why.T, predictions[t] - targets[t]) + dh_next
            dh_raw = (1 - rnn.hs[t] ** 2) * dh
            dL_dWxh += np.dot(dh_raw, rnn.inputs[t].T)
            dL_dWhh += np.dot(dh_raw, rnn.hs[t-1].T)
            dL_dbh += dh_raw
            dh_next = np.dot(rnn.Whh.T, dh_raw)

        # 更新权重和偏差
        rnn.Wxh -= learning_rate * dL_dWxh
        rnn.Whh -= learning_rate * dL_dWhh
        rnn.Why -= learning_rate * dL_dWhy
        rnn.bh -= learning_rate * dL_dbh.reshape(-1, 1)
        rnn.by -= learning_rate * dL_dby

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss / len(input_data)}')

# 在训练后用模型进行预测
predictions = rnn.forward(input_data[0])
predicted_values = np.array([y[0, 0] for y in predictions]).flatten()

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(x_values[time_steps:], derivative_values[time_steps:], label='True Derivative')
plt.plot(x_values[time_steps:], predicted_values, label='Predicted Derivative', linestyle='dashed')
plt.legend()
plt.show()
