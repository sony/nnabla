import os
import glob
import argparse
import numpy as np
import argparse

# nabla imports
import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as nm
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

# args
from config import load_decoder_config, load_encoder_config, load_transformer_config
from utils import MonitorManager, combine_images
from nnabla.utils.image_utils import imresize

# import my models definition
import models
# import my data iterator
import data


def train(encoder_config, transformer_config, decoder_config,
          encoder_netG, transformer_netG, decoder_netG,
          src_celeb_name, trg_celeb_name,
          train_iterator, monitor,
          encoder_param_file, transformer_param_file, decoder_param_file):
    # prepare nn.Variable
    real_img = nn.Variable((1, 3, 256, 256))
    real_bod_map = nn.Variable((1, 15, 64, 64))
    real_bod_map_resize = nn.Variable((1, 15, 256, 256))

    # encoder
    with nn.parameter_scope(encoder_config["model_name"]):
        _, preds = encoder_netG(
            real_img,
            batch_stat=False,
            planes=encoder_config["model"]["planes"],
            output_nc=encoder_config["model"]["output_nc"],
            num_stacks=encoder_config["model"]["num_stacks"],
            activation=encoder_config["model"]["activation"],
        )
    preds.persistent = True
    preds_unlinked = preds.get_unlinked_variable()

    # load parameters of networks
    # with nn.parameter_scope(encoder_config["model_name"]):
    #     nn.load_parameters(encoder_param_file)

    # transformer
    if "star" in transformer_config["dataset_mode"]:

        celeb_name = trg_celeb_name
        if celeb_name == 'Donald_Trump':
            label = np.array([1., 0., 0., 0., 0.])
        elif celeb_name == 'Emmanuel_Macron':
            label = np.array([0., 1., 0., 0., 0.])
        elif celeb_name == 'Jack_Ma':
            label = np.array([0., 0., 1., 0., 0.])
        elif celeb_name == 'Kathleen':
            label = np.array([0., 0., 0., 1., 0.])
        elif celeb_name == 'Theresa_May':
            label = np.array([0., 0., 0., 0., 1.])
        else:
            label = np.array([1., 0., 0., 0., 0.])
        label_trg = nn.Variable.from_numpy_array(
                                        np.reshape(label, (1, 5, 1, 1)))

        # Generate fake image
        with nn.parameter_scope('netG_star'):
            fake_bod_map = transformer_netG(preds_unlinked, label_trg)

        # # load parameters of networks
        # with nn.parameter_scope('netG_star'):
        #     nn.load_parameters(transformer_param_file)
    else:
        # Generator

        with nn.parameter_scope('netG_transformer'):
            with nn.parameter_scope('netG_A2B'):
                fake_bod_map = transformer_netG(
                    preds, test=True, norm_type=transformer_config["norm_type"])

        # with nn.parameter_scope('netG_transformer'):
        #     with nn.parameter_scope('netG_A2B'):
        #         nn.load_parameters(transformer_param_file)

    fake_bod_map.persistent = True
    fake_bod_map_unlinked = fake_bod_map.get_unlinked_variable()

    # decoder
    with nn.parameter_scope('netG_decoder'):
        fake_img = decoder_netG(fake_bod_map_unlinked, test=True)
    fake_img.persistent = True

    # # load parameters of networks
    # with nn.parameter_scope('netG_decoder'):
    #     nn.load_parameters(decoder_param_file)

    monitor_vis = nm.MonitorImage('result',
                                  monitor,
                                  interval=1,
                                  num_images=1,
                                  normalize_method=lambda x: x)

    # train
    num_train_batches = train_iterator.size
    for i in range(num_train_batches):
        _real_img, _real_bod_map, _real_bod_map_resize = train_iterator.next()

        real_img.d = _real_img
        real_bod_map.d = _real_bod_map
        real_bod_map_resize.d = _real_bod_map_resize

        # Generator
        preds.forward(clear_no_need_grad=True)
        # import pdb;pdb.set_trace()

        fake_bod_map.forward(clear_no_need_grad=True)

        fake_img.forward(clear_no_need_grad=True)

        images_to_visualize = [real_img.d,
                               preds.d,
                               fake_bod_map.d,
                               fake_img.d,
                               real_bod_map_resize.d]

        visuals = combine_images(images_to_visualize)
        monitor_vis.add(i, visuals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_config', default=None, type=str)
    parser.add_argument('--transformer_config', default=None, type=str)
    parser.add_argument('--decoder_config', default=None, type=str)

    parser.add_argument('--src_celeb_name', default=None, type=str)
    parser.add_argument('--trg_celeb_name', default=None, type=str)

    parser.add_argument('--info', default=None, type=str)
    args = parser.parse_args()

    encoder_config = load_encoder_config(args.encoder_config)
    transformer_config = load_transformer_config(args.transformer_config)
    decoder_config = load_decoder_config(args.decoder_config)

    src_celeb_name = args.src_celeb_name
    trg_celeb_name = args.trg_celeb_name

    if args.info:
        decoder_config["experiment_name"] += args.info

    #########################
    # Context Setting
    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info(f'Running in {decoder_config["context"]}.')
    ctx = get_extension_context(
        decoder_config["context"], device_id=decoder_config["device_id"])
    nn.set_default_context(ctx)
    #########################

    # Data Loading
    logger.info('Initialing Datasource')

    if decoder_config["ref_dir"]:
        # use pre-calculated heatmaps.
        # Note that you need to run a preprocess script in advance.
        assert os.path.exists(decoder_config["ref_dir"]), \
               f'{decoder_config["ref_dir"]} not found.'
        assert os.path.isfile(os.path.join(decoder_config["ref_dir"],
                                           f'{decoder_config["trg_celeb_name"]}_image.npz')), \
            ".npz file containing image is missing."
        assert os.path.isfile(os.path.join(decoder_config["ref_dir"],
                                           f'{decoder_config["trg_celeb_name"]}_heatmap.npz')), \
            ".npz file containing heatmap is missing."
        assert os.path.isfile(os.path.join(decoder_config["ref_dir"],
                                           f'{decoder_config["trg_celeb_name"]}_resized_heatmap.npz')), \
            ".npz file containing resized heatmap is missing."

        decoder_config["dataset_mode"] = "decoder_ref"
        train_iterator = data.celebv_data_iterator(decoder_config["train_dir"],
                                                   trg_celeb_name=src_celeb_name,
                                                   dataset_mode="decoder_ref",
                                                   batch_size=1,
                                                   shuffle=False,
                                                   with_memory_cache=decoder_config["test"]["with_memory_cache"],
                                                   with_file_cache=decoder_config["test"]["with_file_cache"],
                                                   transform=None,
                                                   ref_dir=decoder_config["ref_dir"],
                                                   mode="train")

    else:
        decoder_config["dataset_mode"] = "decoder"
        train_iterator = data.celebv_data_iterator(decoder_config["train_dir"],
                                                   trg_celeb_name=src_celeb_name,
                                                   dataset_mode="decoder",
                                                   batch_size=1,
                                                   shuffle=False,
                                                   with_memory_cache=decoder_config["test"]["with_memory_cache"],
                                                   with_file_cache=decoder_config["test"]["with_file_cache"],
                                                   mode="train")
    # Encoder
    encoder_netG = models.stacked_hourglass_net

    # Transformer
    if "star" in transformer_config["dataset_mode"]:
        transformer_netG = models.netG_star

    else:
        transformer_netG = {'netG_A2B': models.netG_transformer,
                            'netG_B2A': models.netG_transformer}

    # Decoder
    decoder_netG = models.netG_decoder

    monitor = nm.Monitor(os.path.join(decoder_config["logdir"],
                                      "end2end",
                                      f'{src_celeb_name}2{decoder_config["trg_celeb_name"]}',
                                      decoder_config["experiment_name"]))

    train(encoder_config, transformer_config, decoder_config,
          encoder_netG, transformer_netG, decoder_netG,
          src_celeb_name, trg_celeb_name,
          train_iterator, monitor)


if __name__ == '__main__':
    main()
