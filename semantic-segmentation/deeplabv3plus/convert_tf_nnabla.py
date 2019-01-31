# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow.python import pywrap_tensorflow

import nnabla as nn
import nnabla.parametric_functions as pf

import numpy

from args import get_args



def tf_to_nn_param_map():
    '''map from tensor name to Nnabla default parameter names 
    '''
    return {
    '/xception_module' : '',
    'BatchNorm' : 'bn',
    'moving_mean' : 'mean',
    'moving_variance' : 'var',
    'depthwise_weights' : 'depthwise_conv/W',
    'weights' : 'conv/W',
    '_pointwise' : '/pointwise',
    '_depthwise' : '/depthwise',
    'conv1_1' : '1',
    'conv1_2' : '2',
    'biases' : 'conv/b',
    'logits/semantic' : 'decoder/logits/affine'
    }




def rename_params(param_name):
    '''
       Rename the ckpt tensor names to corresp. Nnabla param names
    '''

    tf_to_nn_dict = tf_to_nn_param_map()
    d_conv = '_depthwise'
    p_conv = '_pointwise'
    
    for key in tf_to_nn_dict:
        if key in param_name:
            param_name = param_name.replace(key, tf_to_nn_dict[key])

    return param_name



def convert(input_file, output_file):
    ''' Convert the input checkpoint file to output hdf5 file
    '''
    parse_tf_ckpt(input_file, output_file)



def parse_tf_ckpt(ckpt_file, h5_file):
    ''' Parse the TF checkpoint file and save as nnabla parameters
    '''
    #Get tensorflow checkpoint reader
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    
    #Loop through each tensor name from the variable to shape map
    for key in sorted(var_to_shape_map):

        #Read tensor values for each tensor name
        weight = reader.get_tensor(key)
        if 'depthwise' in key and weight.ndim == 4:
            weight = numpy.squeeze(weight, axis=3)  
            weight = numpy.transpose(weight, (2,0,1))

        else:
            if(weight.ndim == 4):
                weight = numpy.transpose(weight, (3,2,0,1))

        if 'BatchNorm' in key:
            weight = weight.reshape((1,-1,1,1))

        if 'Momentum' in key or 'ExponentialMovingAverage' in key:
            continue

        key = rename_params(key) 
        

        #Create parameter with the same tensor name and shape
        params = pf.get_parameter_or_create(key, shape=weight.shape)
        params.d = weight

    # Save to a h5 file
    nn.parameter.save_parameters(h5_file)



def main():
    ''' 
    Main
    
    Usage: python convert_tf_nnabla.py --input-ckpt-file=/path to ckpt file --output-nnabla-file=/output .h5 file

    '''

    #Parse the arguments
    args = get_args()

    #convert the input file(.ckpt) to the output file(.h5)
    convert(args.input_ckpt_file, args.output_nnabla_file)



if __name__ == '__main__':
    main()
