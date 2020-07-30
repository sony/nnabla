import os
import yaml
import random
import argparse

import numpy as np

import nnabla as nn
import nnabla.solvers as S
import nnabla.monitor as nm
import nnabla.functions as F
import nnabla.logger as logger

import data
from config import load_encoder_config
from models import stacked_hourglass_net
from utils import MonitorManager, combine_images

from nnabla.ext_utils import get_extension_context


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # may need to set it for nnabla


def train(config, train_iterator, valid_iterator, monitor):

    ################### Graph Construction ####################
    # Training graph
    img, htm = train_iterator.next()
    image = nn.Variable(img.shape)
    heatmap = nn.Variable(htm.shape)

    with nn.parameter_scope(config["model_name"]):
        preds = stacked_hourglass_net(
            image,
            batch_stat=True,
            planes=config["model"]["planes"],
            output_nc=config["model"]["output_nc"],
            num_stacks=config["model"]["num_stacks"],
            activation=config["model"]["activation"],
        )

    if config["finetune"]:
        os.path.isfile(config["finetune"]["param_path"]
                       ), "params file not found."
        with nn.parameter_scope(config["model_name"]):
            nn.load_parameters(config["finetune"]["param_path"])

    # Loss Definition
    if config["loss_name"] == 'mse':
        def loss_func(pred, target): return F.mean(
            F.squared_error(pred, target))
    elif config["loss_name"] == 'bce':
        def loss_func(pred, target): return F.mean(
            F.binary_cross_entropy(pred, target))
    else:
        raise NotImplementedError

    losses = []
    for pred in preds:
        loss_local = loss_func(pred, heatmap)
        loss_local.persistent = True
        losses.append(loss_local)

    loss = nn.Variable()
    loss.d = 0
    for loss_local in losses:
        loss += loss_local

    ################### Setting Solvers ####################
    solver = S.Adam(config["train"]["lr"])
    with nn.parameter_scope(config["model_name"]):
        solver.set_parameters(nn.get_parameters())

    # Validation graph
    img, htm = valid_iterator.next()
    val_image = nn.Variable(img.shape)
    val_heatmap = nn.Variable(htm.shape)

    with nn.parameter_scope(config["model_name"]):
        val_preds = stacked_hourglass_net(
            val_image,
            batch_stat=False,
            planes=config["model"]["planes"],
            output_nc=config["model"]["output_nc"],
            num_stacks=config["model"]["num_stacks"],
            activation=config["model"]["activation"],
        )

    for i in range(len(val_preds)):
        val_preds[i].persistent = True

    # Loss Definition
    val_losses = []
    for pred in val_preds:
        loss_local = loss_func(pred, val_heatmap)
        loss_local.persistent = True
        val_losses.append(loss_local)

    val_loss = nn.Variable()
    val_loss.d = 0
    for loss_local in val_losses:
        val_loss += loss_local

    num_train_batches = train_iterator.size // train_iterator.batch_size + 1
    num_valid_batches = valid_iterator.size // valid_iterator.batch_size + 1

    ################### Create Monitors ####################
    monitors_train_dict = {'loss_total': loss}

    for i in range(len(losses)):
        monitors_train_dict.update({f'loss_{i}': losses[i]})

    monitors_val_dict = {'val_loss_total': val_loss}

    for i in range(len(val_losses)):
        monitors_val_dict.update({f'val_loss_{i}': val_losses[i]})

    monitors_train = MonitorManager(
        monitors_train_dict, monitor, interval=config["monitor"]["interval"]*num_train_batches)
    monitors_val = MonitorManager(
        monitors_val_dict, monitor, interval=config["monitor"]["interval"]*num_valid_batches)
    monitor_time = nm.MonitorTimeElapsed(
        'time', monitor, interval=config["monitor"]["interval"]*num_train_batches)
    monitor_vis = nm.MonitorImage(
        'result', monitor, interval=1, num_images=4, normalize_method=lambda x: x)
    monitor_vis_val = nm.MonitorImage(
        'result_val', monitor, interval=1, num_images=4, normalize_method=lambda x: x)

    os.mkdir(os.path.join(monitor._save_path, 'model'))

    # Dump training information
    with open(os.path.join(monitor._save_path, "training_info.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml.dump(config))

    # Training
    best_epoch = 0
    best_val_loss = np.inf

    for e in range(config["train"]["epochs"]):
        watch_val_loss = 0

        # training loop
        for i in range(num_train_batches):
            image.d, heatmap.d = train_iterator.next()

            solver.zero_grad()
            loss.forward()
            loss.backward(clear_buffer=True)
            solver.weight_decay(config["train"]["weight_decay"])
            solver.update()

            monitors_train.add(e*num_train_batches + i)
            monitor_time.add(e*num_train_batches + i)

        # validation loop
        for i in range(num_valid_batches):
            val_image.d, val_heatmap.d = valid_iterator.next()
            val_loss.forward(clear_buffer=True)
            monitors_val.add(e*num_valid_batches + i)

            watch_val_loss += val_loss.d.copy()

        watch_val_loss /= num_valid_batches

        # visualization
        visuals = combine_images([image.d, preds[0].d, preds[1].d, heatmap.d])
        monitor_vis.add(e, visuals)

        visuals_val = combine_images(
            [val_image.d, val_preds[0].d, val_preds[1].d, val_heatmap.d])
        monitor_vis_val.add(e, visuals_val)

        # update best result and save weights if updated
        if best_val_loss > watch_val_loss or e % config["monitor"]["save_interval"] == 0:
            best_val_loss = watch_val_loss
            best_epoch = e
            save_path = os.path.join(
                monitor._save_path, 'model/model_epoch-{}.h5'.format(e))
            with nn.parameter_scope(config["model_name"]):
                nn.save_parameters(save_path)

    # save the last parameters as well
    save_path = os.path.join(
        monitor._save_path, 'model/model_epoch-{}.h5'.format(e))
    with nn.parameter_scope(config["model_name"]):
        nn.save_parameters(save_path)

    logger.info(f'Best Epoch: {best_epoch}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--info', default=None, type=str)
    args = parser.parse_args()

    config = load_encoder_config(args.config)
    if args.info:
        config["experiment_name"] += args.info

    #########################
    # Context Setting
    logger.info(f'Running in {config["context"]}.')
    ctx = get_extension_context(
        config["context"], device_id=config["device_id"])
    nn.set_default_context(ctx)
    #########################

    seed_everything(config["seed"])
    if config["train"]["augmentation"]:
        import albumentations as A
        transform = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=10/256., scale_limit=0.03, rotate_limit=5, always_apply=True),
            A.HorizontalFlip(p=0.5),
        ])
    else:
        transform = None

    train_iterator = data.wflw_data_iterator(data_dir=config["path"]["data_dir"],
                                             dataset_mode=config["dataset_mode"],
                                             mode="train", use_reference=config["use_reference"],
                                             batch_size=config["train"]["batch_size"],
                                             transform=transform, shuffle=True, rng=None,
                                             with_memory_cache=config["train"]["with_memory_cache"],
                                             with_file_cache=config["train"]["with_file_cache"])

    valid_iterator = data.wflw_data_iterator(data_dir=config["path"]["data_dir"],
                                             dataset_mode=config["dataset_mode"], mode="test",
                                             use_reference=config["use_reference"],
                                             batch_size=config["test"]["batch_size"],
                                             transform=None, shuffle=False, rng=None,
                                             with_memory_cache=config["test"]["with_memory_cache"],
                                             with_file_cache=config["test"]["with_file_cache"])

    # make new dir into config,output_dir for monitor
    monitor = nm.Monitor(os.path.join(
        config["logdir"], config["dataset_mode"], config["experiment_name"]))

    # training model
    train(config=config, train_iterator=train_iterator,
          valid_iterator=valid_iterator, monitor=monitor)


if __name__ == '__main__':
    main()
