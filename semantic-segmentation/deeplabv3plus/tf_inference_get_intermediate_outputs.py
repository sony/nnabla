import tensorflow as tf
import numpy as np
import cv2

image = cv2.imread("cat1.jpeg")

new_saver = tf.train.import_meta_graph('model.ckpt-200000.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    out = graph.get_tensor_by_name('xception_65/entry_flow/block3/unit_1/xception_module/separable_conv3_pointwise/weights:0')
    x = tf.placeholder(tf.float32, shape=[None, 168, 301, 3])
    print(sess.run(out, feed_dict={x: [np.asarray(image)]}))

    #Print the output of an intermediate layer
    #print(sess.run('xception_41/entry_flow/block1/unit_1/xception_module/separable_conv3_pointwise/weights:0'))
    #print(sess.run('xception_41/entry_flow/block2/unit_1/xception_module/separable_conv3_pointwise/weights:0'))
    #print(sess.run('xception_41/entry_flow/block3/unit_1/xception_module/separable_conv3_pointwise/weights:0'))

