
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
