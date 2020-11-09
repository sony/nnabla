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

import os
import glob
import numpy as np
import nnabla as nn
import nnabla.logger as logger

from tqdm import tqdm
from scipy import linalg
from .im2ndarray import im2ndarray
from .inceptionv3 import construct_inceptionv3
from nnabla.ext_utils import get_extension_context


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, nargs=2,
                        help='paths to the directory containing real/fake images, \
                              or the text file listing real/fake images, \
                              or .npz file containing statistics of real/fake images. \
                              Expects the former is the real, and the latter is the fake.')
    parser.add_argument('--params-path',
                        default=os.path.dirname(os.path.abspath(__file__)) +
                        '/original_inception_v3.h5', type=str,
                        help='path to the weight file (.h5).')
    parser.add_argument('--save-stats', action="store_true",
                        help='if specified, save image distribution statistics, respectively. \
                        if .npz file is already given, the process is skipped.')
    parser.add_argument('--saved-filenames', type=str, nargs=2, default=["", ""],
                        help='names of the saved stats files.')
    parser.add_argument('--device-id', '-d', default=0, type=int,
                        help='Device ID.')
    parser.add_argument('--context', '-c', default="cudnn", type=str,
                        help='Backend name.')
    parser.add_argument('--batch-size', '-b', default=16, type=int,
                        help='batch-size. automatically adjusted. see the code.')
    return parser.parse_args()


def calculate_fid(mu1, mu2, sigma1, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    code from https://github.com/bioinf-jku/TTUR.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        logger.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_stats(feat: np.ndarray):
    """Compute mean and covariance of given features.
        Args:
            feat (np.ndarray): extracted features. shape: (N, 2048)

        Returns:
            mu (np.ndarray): mean feature. shape: (2048,)
            cov (np.ndarray): covariance of features. (2048, 2048)
    """
    mu = np.mean(feat, axis=0)
    cov = np.cov(feat, rowvar=False)
    return mu, cov


def get_features(input_images: nn.NdArray):
    """Extract image features using Inception v3.
        Args:
            input_images (nn.NdArray):  NdArrays representing images.
                                        Shape must be (B, 3, 299, 299).
                                        Must be pre-normalized, i.e. its values must lie in [-1., +1.]

        Returns:
            feature (nn.NdArray): Image feature. shape: (B, 2048)
    """
    feature = construct_inceptionv3(input_images)
    return feature


def get_all_features_on_imagepaths(image_paths: list, batch_size: int):
    """Extract all the given images' feature.
        Args:
            image_paths (list): list of image file's paths.
            batch_size (int): batch size.

        Returns:
            all_feat (np.ndarray): extracted (N) images' feature. shape: (N, 2048)
    """
    print("loading images...")
    num_images = len(image_paths)
    if num_images < 9999:
        logger.warning(f"only {num_images} images found. It may produce inaccurate FID score.")
    num_loop, num_remainder = divmod(num_images, batch_size)
    batched_images = image_paths[:-num_remainder]
    rest_image_paths = image_paths[-num_remainder:]

    pbar = tqdm(total=num_images)
    if batch_size > 1 and num_remainder != 0:
        images = im2ndarray(rest_image_paths, imsize=(299, 299))
        feature = get_features(images)
        all_feat = feature.data
        pbar.update(num_remainder)
    else:
        # when batch_size = 1
        all_feat = np.zeros((0, 2048))
        batched_images = rest_image_paths

    for i in range(num_loop):
        image_paths = batched_images[i*batch_size:(i+1)*batch_size]
        images = im2ndarray(image_paths, imsize=(299, 299))
        feature = get_features(images)
        all_feat = np.concatenate([all_feat, feature.data], axis=0)
        pbar.update(batch_size)

    return all_feat


def get_statistics_from_given_path(path, batch_size):
    """Handling the path and get the statistics required for FID calculation. 
        Args:
            path (str): path to the directory containing images,
                        or the text file listing the image files. 
            batch_size (int): batch size.

        Returns:
            mu (np.ndarray): mean feature. shape: (2048,)
            sigma (np.ndarray): covariance of features. shape: (2048, 2048)
    """
    ext = os.path.splitext(path)[-1]
    if ext == ".npz":
        assert os.path.isfile(path), "specified .npz file is not found."
        data = np.load(path)
        assert "mu" in data.files and "sigma" in data.files, "mu and/or sigma not found in the loaded .npz file."
        mu = data["mu"]
        sigma = data["sigma"]
        return mu, sigma

    elif ext == "":
        # path points to a directory
        assert os.path.isdir(path), "specified directory is not found."
        image_paths = glob.glob(f"{path}/*.png") + glob.glob(f"{path}/*.jpg")
        # of simply glob.glob(f"{path}/*")?
    elif ext == ".txt":
        assert os.path.isfile(path), "specified file is not found."
        with open(path, "r") as f:
            image_paths = [_.rstrip("\n") for _ in f.readlines()]
    else:
        raise RuntimeError(f"Invalid path: {path}")

    feature = get_all_features_on_imagepaths(image_paths, batch_size)
    mu, sigma = get_stats(feature)

    return mu, sigma


def save_statistics(save_stat_path, image_file_path, mu, sigma):
    """Save computed statistics as .npz format.
        Args:
            save_stat_path (str): A name of .npz file to be saved. Can be empty string,
                                  but then automatically uses the part of image_file_path.
            image_file_path (str): a directory containing image files or 
                                   a text file listing image files.
            mu (np.ndarray): computed mean feature.
            sigma (np.ndarray):  computed covariance of features.
    """
    if save_stat_path:
        filename = save_stat_path
    else:
        filename = os.path.splitext(image_file_path.split("/")[-1])[0]
    np.savez_compressed(filename, mu=mu, sigma=sigma)
    print(f"Saved {filename}.npz")


def load_parameters(params_path):
    if not os.path.isfile(params_path):
        from nnabla.utils.download import download
        url = "https://nnabla.org/pretrained-models/nnabla-examples/eval_metrics/inceptions/original_inception_v3.h5"
        download(url, params_path, False)
    nn.load_parameters(params_path)


def main():

    args = get_args()
    # Context
    ctx = get_extension_context(args.context, device_id=args.device_id)
    nn.set_default_context(ctx)
    batch_size = args.batch_size
    save_stats = args.save_stats

    paths = args.path

    load_parameters(args.params_path)  # overwrite

    print("Computing statistics...")
    mu1, sigma1 = get_statistics_from_given_path(paths[0], batch_size)
    if save_stats:
        save_statistics(args.saved_filenames[0], paths[0], mu1, sigma1)

    mu2, sigma2 = get_statistics_from_given_path(paths[1], batch_size)
    if save_stats:
        save_statistics(args.saved_filenames[1], paths[1], mu2, sigma2)

    score = calculate_fid(mu1, mu2, sigma1, sigma2)

    print(f"Image Set 1: {paths[0]}")
    print(f"Image Set 2: {paths[1]}")
    print(f"batch size: {batch_size}")
    print(f"Frechet Inception Distance: {score:.5f}")


if __name__ == '__main__':
    main()
