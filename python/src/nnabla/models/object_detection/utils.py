from nnabla.utils.image_utils import imread, imresize, imsave
from nnabla.models.object_detection.draw_utils import DrawBoundingBoxes
import numpy as np


def draw_bounding_boxes(img, bboxes, im_w, im_h, names, colors, thresh):
    '''
    The transformed cordinates are futher used to draw bounding boxes for the detected objects.

    Args:

            img (Variable) : 
                 Input image, type ``nnabla Variable``
            bboxes : 
                 Transformed bounding box co-orinates from the model.
            im_w : 
                 Width of original input image.
            im_h :
                 Height of original input image.
            names : 
                 Name of categories in the dataset
            colors : 
                 Colors for bunding boxes
    '''
    draw = DrawBoundingBoxes(img, colors)
    for bb in bboxes:
        x, y, w, h = bb[:4]
        dw = w / 2.
        dh = h / 2.
        x0 = int(np.clip(x - dw, 0, im_w))
        y0 = int(np.clip(y - dh, 0, im_h))
        x1 = int(np.clip(x + dw, 0, im_w))
        y1 = int(np.clip(y + dh, 0, im_h))
        det_ind = np.where(bb[5:] > thresh)[0]
        if len(det_ind) == 0:
            continue
        prob = bb[5 + det_ind]
        label = ', '.join("{}: {:.2f}%".format(
            names[det_ind[j]], prob[j] * 100) for j in range(len(det_ind)))
        print("[INFO] {}".format(label))
        draw.draw((x0, y0, x1, y1), det_ind[0], label)
    return draw.get()


def apply_inverse_letterbox_coordinate_transform(bboxes, im_w, im_h, letterbox_w, letterbox_h):
    '''
    The predicted bounding box co-ordinates from the model are not according to original image but the pre-processed image. This function transforms the co-orinates
    according to original image by applying inverse letterbox co-rdinate trasforms mathematically.

    Args:

            bboxes (Variable) : 
                 The bounding box co-ordinates predicted from the model.
            im_w : 
                 Width of original input image.
            im_h :
                 Height of original input image.
    '''
    for bb in bboxes:
        x, y, w, h = bb[:4]
        x1 = (x - (1 - letterbox_w) / 2.) / letterbox_w * im_w
        y1 = (y - (1 - letterbox_h) / 2.) / letterbox_h * im_h
        w1 = w * im_w / letterbox_w
        h1 = h * im_h / letterbox_h
        bb[:4] = x1, y1, w1, h1
    return bboxes


def letterbox(img_orig, h, w):
    '''
    Input image is pre-processed before passing it to the network in YoloV2. This function applies the pre-processing to input image.

    Args:

            img_orig (Variable) : 
                 Input image, type ``nnabla Variable``
            w : 
                 Desired width of output image after pre-processing. Should be a multiple of 32.
            h : 
                 Desired height of output image after pre-processing. Should be a multiple of 32.
    '''
    im_h, im_w, _ = img_orig.shape
    if (w * 1.0 / im_w) < (h * 1. / im_h):
        new_w = w
        new_h = int((im_h * w) / im_w)
    else:
        new_h = h
        new_w = int((im_w * h) / im_h)

    patch = imresize(img_orig, (new_w, new_h))
    img = np.ones((h, w, 3), np.float32) * 0.5
    # resize
    x0 = int((w - new_w) / 2)
    y0 = int((h - new_h) / 2)
    img[y0:y0 + new_h, x0:x0 + new_w] = patch
    return img, new_w, new_h, im_h, im_w
