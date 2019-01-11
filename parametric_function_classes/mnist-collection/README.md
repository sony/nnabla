# PyTorch-like function in NNabla

In NNabla, parametric functions (e.g., PF.convolution) are defined like TensorFlow V1.x, i.e., trainable parameters are managed in a global scope of a process by using a dictionary of a name of parameter to a trainable parameter and the scope context. It is intuitively straightforward when writing a code while a bit complicated to manage trainable parameters since trainable parameters are not managed in a local scope of e.g., a class but globally, which sometimes hard to see a whole picture of a neural network. If trainable parameters are held in a class like PyTorch or Chainer, it is very easy to see a whole picture of a network while being redundant representation of a network when writing a code since we have to write two lines about a parametric function to be used i.e., in the *\_\_init\_\_* method and *\_\_call\_\_* (or *forward*) method of a class. There are pros and cons for each, but people seem to like PyTorch-like parametric function definition.

Currently, the following NNabla's parametric function classes are defined:

- Convolution (Conv1d, Conv2d, Conv3d, ConvNd)
- Deconvolution (Deconv1d, Deconv2d, Deconv3d, DeconvNd)
- Affine (Linear)
- Embed
- BatchNorm (BatchNorm1d, BatchNorm2d, BatchNorm3d)


# Usage

```python
...
import nnabla.experimental.parametric_function_classes as PFC
...

...
class ResUnit(PFC.Module):
    def __init__(self, inmaps=64, outmaps=64):
        self.conv0 = PFC.Conv2d(inmaps, inmaps // 2, (1, 1), (1, 1), (1, 1), with_bias=False)
        self.bn0 = PFC.BatchNorm2d(inmaps // 2)
        self.conv1 = PFC.Conv2d(inmaps // 2, inmaps // 2, (3, 3), (1, 1), (1, 1), with_bias=False)
        self.bn1 = PFC.BatchNorm2d(inmaps // 2)
        self.conv2 = PFC.Conv2d(maps // 2, outmaps, (1, 1), (1, 1), (1, 1), with_bias=False)
        self.bn2 = PFC.BatchNorm2d(outmaps)
        self.act = F.relu

        self.shortcut_func = False
        if inmaps != outmaps:
            self.shortcut_func = True
            self.shortcut_conv = PFC.Conv2d(inmaps, outmaps, (3, 3), (1, 1), (1, 1), with_bias=False)
            self.shortcut_bn = PFC.BatchNorm2d(outmaps)

    def __call__(self, x, test=False):
        s = x
        h = x
        h = self.act(self.bn0(self.conv0(h), test))
        h = self.act(self.bn1(self.conv1(h), test))
        h = self.bn2(self.conv1(h), test)
        if self.shortcut_func:
          s = self.shortcut_conv(s)
          s = self.shortcut_bn(s)
        h = self.act(h + s)
        return h
...
```

It is easier to see the [main.py](./main.py) than describing how to use in details here.


# Training Example

[main.py](./main.py) includes almost everything to train NN by using *parametric function classes*, run the script like, 


```bash
python main.py
```

