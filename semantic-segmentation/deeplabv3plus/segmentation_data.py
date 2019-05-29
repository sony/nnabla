from nnabla.utils.data_iterator import data_iterator_simple
import cv2
import image_preprocess
import numpy as np
import time


def data_iterator_segmentation(num_examples, batch_size, image_path_file, label_path_file, rng=None, target_width=513, target_height=513, train=True):

    image_paths = load_paths(image_path_file)
    label_paths = load_paths(label_path_file)

    def image_label_load_func(i):
        '''
        Returns:
            image: c x h x w array
            label: c x h x w array
            mask: c x h x w array
        '''

        img = cv2.imread(image_paths[i]).astype('float32')
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        if 'png' in label_paths[i]:
            lab = imageio.imread(
                label_paths[i], as_gray=False, pilmode="RGB").astype('int32')
        else:
            lab = np.load(label_paths[i], allow_pickle=True).astype('int32')
        if lab.ndim == 2:
            lab = lab[..., None]
        # Compute image preprocessing time
        #t = time.time()
        img, lab, mask = image_preprocess.preprocess_image_and_label(
            img, lab, target_width, target_height, train=train)
        #elapsed = time.time() - t

        return np.rollaxis(img, 2), np.rollaxis(lab, 2), np.rollaxis(mask, 2)

    return data_iterator_simple(image_label_load_func, num_examples, batch_size, shuffle=True, rng=rng, with_file_cache=False, with_memory_cache=False)


def load_paths(path_file):
    text_file = open(path_file, "r")
    lines = [line[:-1] for line in text_file]

    return lines
