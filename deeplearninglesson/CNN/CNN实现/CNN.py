
"""
引入了MNIST手写数字数据集。
了解了 Conv 图层，该图层将过滤器与图像卷积以生成更有用的输出。
谈到了池化层，它可以帮助修剪除最有用的功能之外的所有内容。
实现了一个 Softmax 层，因此我们可以使用交叉熵损失。
CNN的实现

"""
import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

# 加载 MNIST 测试数据集
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

# 初始化卷积层、池化层和 Softmax 层
conv = Conv3x3(8)  # 输入：28x28x1 -> 输出：26x26x8
pool = MaxPool2()  # 输入：26x26x8 -> 输出：13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 输入：13x13x8 -> 输出：10


def forward(image, label):
    '''
  完成 CNN 的前向传播，计算准确率和交叉熵损失。
  - image 是一个二维的 numpy 数组
  - label 是一个数字
  '''
    # 将图像从 [0, 255] 转换为 [-0.5, 0.5]，以便更容易处理
    normalized_image = (image / 255) - 0.5

    # 前向传播
    out = conv.forward(normalized_image)
    out = pool.forward(out)
    out = softmax.forward(out)

    # 计算交叉熵损失和准确率
    loss = -np.log(out[label])
    accuracy = 1 if np.argmax(out) == label else 0

    return out, loss, accuracy


print('MNIST CNN initialized!')

total_loss = 0
total_correct = 0

for i, (image, label) in enumerate(zip(test_images, test_labels)):
    # 进行前向传播
    _, current_loss, current_accuracy = forward(image, label)

    # 累计损失和正确分类数
    total_loss += current_loss
    total_correct += current_accuracy

    # 每100步打印一次统计信息
    if (i + 1) % 100 == 0:
        average_loss = total_loss / 100
        accuracy_percentage = (total_correct / 100) * 100
        print(
            f'[Step {i + 1}] Past 100 steps: Average Loss {average_loss:.3f} | Accuracy: {accuracy_percentage:.2f}%'
        )
        total_loss = 0
        total_correct = 0
