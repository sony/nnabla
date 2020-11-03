import nnabla as nn 
import nnabla.functions as F 
import nnabla.parametric_functions as PF

import numpy as np 

class PixelCNN(object):
    
    def __init__(self, channels=64, kernel_size=7, masked_conv_layers=8):
        self.channels = channels
        self.kernel_size = (kernel_size, kernel_size)
        self.masked_conv_layers = masked_conv_layers

    def causal_mask(self, width, height, starting_point):
        row_grid, col_grid = np.meshgrid(
            np.arange(width), np.arange(height), indexing='ij')
        mask = np.logical_or(
            row_grid < starting_point[0],
            np.logical_and(row_grid == starting_point[0], col_grid <= starting_point[1]))
        return mask

    def conv_mask(self, width, height, include_center=0):
        return 1.0 * self.causal_mask(width, height, starting_point=(width//2, height//2 + include_center - 1))

    def mask_type_A(self, W):
        width = W.shape[2]
        height = W.shape[3]
        mask = self.conv_mask(width, height, 0)
        mask = np.expand_dims(mask, 0)
        mask = nn.Variable.from_numpy_array(np.broadcast_to(mask, W.shape))
        W = mask*W
        return W

    def mask_type_B(self, W):
        width = W.shape[2]
        height = W.shape[3]
        mask = self.conv_mask(width, height, 1)
        mask = np.expand_dims(mask, 0)
        mask = nn.Variable.from_numpy_array(np.broadcast_to(mask, W.shape))
        W = mask*W
        return W

    def __call__(self, img):

        with nn.parameter_scope('pixel_cnn'):
            for i in range(self.masked_conv_layers):
                if i == 0:
                    out = PF.convolution(img, self.channels, self.kernel_size, stride=(1, 1), pad=(
                        3, 3), apply_w=self.mask_type_A, name='masked_conv_{}'.format(i))
                else:
                    out = PF.convolution(img, self.channels, self.kernel_size, stride=(1, 1), pad=(
                        3, 3), apply_w=self.mask_type_B, name='masked_conv_{}'.format(i))
                out = PF.batch_normalization(out, name='bn_{}'.format(i))
                out = F.relu(out)

            out = PF.convolution(out, 256, (1, 1), name='conv')

        return out