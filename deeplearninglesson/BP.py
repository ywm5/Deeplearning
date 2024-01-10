import math
class Tensor:
  def __init__(self, value):
    assert isinstance(value, (float, int)), f'int and float only, but value={value} is a {type(value)}'
    self.value = value
    self.parent_nodes = []
    self.parent_node_gradient_functions = []
    self.gradient = 0.0

  def __add__(self, other):
    result = Tensor(self.value + other.value)
    result.parent_nodes.append(self)
    result.parent_nodes.append(other)
    result.parent_node_gradient_functions.append(lambda g : g)
    result.parent_node_gradient_functions.append(lambda g : g)
    return result

  def __sub__(self, other):
    result = Tensor(self.value - other.value)
    result.parent_nodes.append(self)
    result.parent_nodes.append(other)
    result.parent_node_gradient_functions.append(lambda g : g)
    result.parent_node_gradient_functions.append(lambda g : -1.0 * g)
    return result

  def __mul__(self, other):
    result = Tensor(self.value * other.value)
    result.parent_nodes.append(self)
    result.parent_nodes.append(other)
    result.parent_node_gradient_functions.append(lambda g : other.value * g)
    result.parent_node_gradient_functions.append(lambda g : self.value * g)
    return result

  def __div__(self, other):
    result = Tensor(self.value / other.value)
    result.parent_nodes.append(self)
    result.parent_nodes.append(other)
    result.parent_node_gradient_functions.append(lambda g : 1.0 / other.value * g)
    result.parent_node_gradient_functions.append(lambda g : self.value * g)
    return result

  def backward(self, g=None):
    g = g or 1.0
    self.gradient += g
    for parent, grad_func in zip(self.parent_nodes, self.parent_node_gradient_functions):
      parent.backward(grad_func(g))

  def reset_gradient(self, back=True):
    self.gradient = 0.0
    for parent in self.parent_nodes:
      parent.reset_gradient(back)
import math
import random
# y = a * x + b
real_a = 3.14
real_b = -17.99
# generate dataset
N = 100
Ys = []
Xs = []
for i in range(N):
  x = (random.random() - 0.5) * 2  # (-1, 1)
  y = real_a * x + real_b
  y += (random.random() - 0.5) * 0.01 # add random error
  Xs.append(x)
  Ys.append(y)

# learning
epochs = 100
lr = 0.001
A = Tensor(random.random())
B = Tensor(random.random())
for epoch in range(epochs):
  epoch_loss = 0.0
  for i in range(N):
    # forward pass, compute the loss
    x = Xs[i]
    y_true = Ys[i]
    X = Tensor(x)
    Y_TRUE = Tensor(y_true)
    Y_PRED = A * X + B
    LOSS = (Y_TRUE - Y_PRED) * (Y_TRUE - Y_PRED)
    epoch_loss += LOSS.value
    # backward pass, compute the LOSS gradients w.r.t A and B
    LOSS.reset_gradient()
    LOSS.backward()
    # apply SGD
    A.value -= lr * A.gradient
    B.value -= lr * B.gradient

  print(f'epoch={epoch}/{epochs}, loss={epoch_loss/N}, A={A.value}, B={B.value}')


print(f'real_a={real_a}, real_b={real_b}')
print(f'learned_a={A.value:.2f}, learned_b={B.value:.2f}')

assert math.isclose(A.value, real_a, rel_tol=1e-02) # Python 3.5 or newer
assert math.isclose(B.value, real_b, rel_tol=1e-02)