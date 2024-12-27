import os, gzip, tarfile, pickle
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch

def fetch_mnist(tensors=False):
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"   # http://yann.lecun.com/exdb/mnist/ lacks https
  X_train = parse(fetch(f"{BASE_URL}train-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse(fetch(f"{BASE_URL}train-labels-idx1-ubyte.gz"))[8:].astype(np.int8)
  X_test = parse(fetch(f"{BASE_URL}t10k-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse(fetch(f"{BASE_URL}t10k-labels-idx1-ubyte.gz"))[8:].astype(np.int8)
  if tensors: return Tensor(X_train).reshape(-1, 1, 28, 28), Tensor(Y_train), Tensor(X_test).reshape(-1, 1, 28, 28), Tensor(Y_test)
  else: return X_train, Y_train, X_test, Y_test

from tinygrad.tensor import Tensor
from tinygrad.helpers import CI, trange
from tinygrad.engine.jit import TinyJit


def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=lambda out,y: out.sparse_categorical_crossentropy(y),
        transform=lambda x: x, target_transform=lambda x: x, noloss=False, allow_jit=True):

  def train_step(x, y):
    # network
    out = model.forward(x) if hasattr(model, 'forward') else model(x)
    loss = lossfn(out, y)
    optim.zero_grad()
    loss.backward()
    if noloss: del loss
    optim.step()
    if noloss: return (None, None)
    cat = out.argmax(axis=-1)
    accuracy = (cat == y).mean()
    return loss.realize(), accuracy.realize()

  if allow_jit: train_step = TinyJit(train_step)

  with Tensor.train():
    losses, accuracies = [], []
    for i in (t := trange(steps, disable=CI)):
      samp = np.random.randint(0, X_train.shape[0], size=(BS))
      x = Tensor(transform(X_train[samp]), requires_grad=False)
      y = Tensor(target_transform(Y_train[samp]))
      loss, accuracy = train_step(x, y)
      # printing
      if not noloss:
        loss, accuracy = loss.numpy(), accuracy.numpy()
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
  return [losses, accuracies]


def evaluate(model, X_test, Y_test, num_classes=None, BS=128, return_predict=False, transform=lambda x: x,
             target_transform=lambda y: y):
  Tensor.training = False
  def numpy_eval(Y_test, num_classes):
    Y_test_preds_out = np.zeros(list(Y_test.shape)+[num_classes])
    for i in trange((len(Y_test)-1)//BS+1, disable=CI):
      x = Tensor(transform(X_test[i*BS:(i+1)*BS]))
      out = model.forward(x) if hasattr(model, 'forward') else model(x)
      Y_test_preds_out[i*BS:(i+1)*BS] = out.numpy()
    Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
    Y_test = target_transform(Y_test)
    return (Y_test == Y_test_preds).mean(), Y_test_preds

  if num_classes is None: num_classes = Y_test.max().astype(int)+1
  acc, Y_test_pred = numpy_eval(Y_test, num_classes)
  print("test set accuracy is %f" % acc)
  return (acc, Y_test_pred) if return_predict else acc

import unittest
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.helpers import CI
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim, BatchNorm2d

# load the mnist dataset
X_train, Y_train, X_test, Y_test = fetch_mnist()

# create a model
class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor.scaled_uniform(784, 128)
    self.l2 = Tensor.scaled_uniform(128, 10)

  def parameters(self):
    return get_parameters(self)

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2)

# create a model with a conv layer
class TinyConvNet:
  def __init__(self, has_batchnorm=False):
    # https://keras.io/examples/vision/mnist_convnet/
    conv = 3
    #inter_chan, out_chan = 32, 64
    inter_chan, out_chan = 8, 16   # for speed
    self.c1 = Tensor.scaled_uniform(inter_chan,1,conv,conv)
    self.c2 = Tensor.scaled_uniform(out_chan,inter_chan,conv,conv)
    self.l1 = Tensor.scaled_uniform(out_chan*5*5, 10)
    if has_batchnorm:
      self.bn1 = BatchNorm2d(inter_chan)
      self.bn2 = BatchNorm2d(out_chan)
    else:
      self.bn1, self.bn2 = lambda x: x, lambda x: x

  def parameters(self):
    return get_parameters(self)

  def forward(self, x:Tensor):
    x = x.reshape(shape=(-1, 1, 28, 28)) # hacks
    x = self.bn1(x.conv2d(self.c1)).relu().max_pool2d()
    x = self.bn2(x.conv2d(self.c2)).relu().max_pool2d()
    x = x.reshape(shape=[x.shape[0], -1])
    return x.dot(self.l1)

@unittest.skipIf(CI and Device.DEFAULT == "CLANG", "slow")
class TestMNIST(unittest.TestCase):
  def test_sgd_onestep(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, BS=69, steps=1)
    for p in model.parameters(): p.realize()

  def test_sgd_threestep(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, BS=69, steps=3)

  def test_sgd_sixstep(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, BS=69, steps=6, noloss=True)

  def test_adam_onestep(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, BS=69, steps=1)
    for p in model.parameters(): p.realize()

  def test_adam_threestep(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, BS=69, steps=3)

  def test_conv_onestep(self):
    np.random.seed(1337)
    model = TinyConvNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, BS=69, steps=1, noloss=True)
    for p in model.parameters(): p.realize()

  def test_conv(self):
    np.random.seed(1337)
    model = TinyConvNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, steps=100)
    print("i know 2")
    assert evaluate(model, X_test, Y_test) > 0.93   # torch gets 0.9415 sometimes

  def test_conv_with_bn(self):
    np.random.seed(1337)
    model = TinyConvNet(has_batchnorm=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.003)
    train(model, X_train, Y_train, optimizer, steps=200)
    print("i know")
    assert evaluate(model, X_test, Y_test) > 0.94

  def test_sgd(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, steps=600)
    assert evaluate(model, X_test, Y_test) > 0.94   # CPU gets 0.9494 sometimes

if __name__ == '__main__':
  unittest.main()