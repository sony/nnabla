# Utilities for nNabla exampls

NEU, nnabla examples utils, provides a bunch of reusable components for writing training and inference scripts in nnabla-examples. Please note that this package is not organized very well and not stable so far, the API might change without any notice.

## How to use

This package is not provided for package managers such as pip and conda so far. You have to set a python path to this folder to use `neu` package.

We usually set a path to this folder at a utils package under each training example folder when you import it. See [Pix2PixHD/utils](../GANs/pix2pixHD/utils/__init__.py) for an example.


# Misc
## Loading Intermediate h5 to nnp 
The below written code demonstrate how the intermediate h5 file can be used for all the examples present in the NNabla-examples repository. When you run the training script, the network file(.nnp) will be saved before the training begins to obatin the architecture of the network even though weights and biases are randomly initialized. The network parameters(.h5) will also be saved. Using these two files you can get the network along with its parameters to do inference.

## Steps to load the h5 file to nnp
* Load the nnp file at epoch 0.
* Load the desired .h5 file.
* Use nnp.get_network_names() to fetch the network name. 
* Create a variable graph of the network by name.
* Load the inputs to the input variable.
* Execute the network.


### Demonstration of the code 
```
import nnabla as nn
from nnabla.utils.image_utils import imread
from nnabla.utils.nnp_graph import NnpLoader
nn.clear_parameters()
nn.load_parameters('Path to saved h5 file')
nnp = NnpLoader('Path to saved nnp file')
img = imread('Path to image file')
net = nnp.get_network('name of the network', batch_size=1)
y = net.outputs['y']
x = net.inputs['x']
x.d = img
y.forward(clear_buffer=True)
```
