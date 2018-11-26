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

from itertools import chain
import glob
import json
import os
import shutil
import yaml
import string
import random

_job_status = {}
_outdir = '.'
random_num = 8


def utcnow_timestamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).timestamp()


def init(args):
    global _outdir
    _outdir = args.outdir

    global _job_status
    status_json_path = os.path.join(_outdir, 'status.json')
    if os.path.exists(status_json_path):
        with open(status_json_path, 'r') as f:
            _job_status = json.load(f)
    else:
        _job_status = {
            'type': args.func.__name__.rsplit('_', 1)[0],
            'start_time': utcnow_timestamp(),
        }

    # In case of Resume, status.json and monitoring_report.yml should be conflicted, then
    # if  monitoring_report.yml is already exists, read and synchronize with it.
    monitoring_report_path = os.path.join(_outdir, 'monitoring_report.yml')
    if os.path.exists(monitoring_report_path):
        with open(monitoring_report_path, 'r') as f:
            data = yaml.load(f)
        # Because the key in monitoring_report.yml is int(epoch - 1),
        # convert it to str(epoch) as same as in status.json.
        # And calculate  best and last also.
        mon_rep = {}
        best_epoch = 0
        best_valid_error = 1.0
        last_train_error = None
        last_valid_error = None
        for k in sorted(data.keys()):
            mon_rep[str(k + 1)] = data[k]
            if 'train_error' in data[k]:
                last_train_error = data[k]['train_error']
            if 'valid_error' in data[k]:
                last_valid_error = data[k]['valid_error']
                if data[k]['valid_error'] < best_valid_error:
                    best_epoch = k + 1
                    best_valid_error = data[k]['valid_error']
        _job_status['monitoring_report'] = mon_rep

        if best_epoch:
            _job_status['best'] = {
                'epoch': best_epoch,
                'valid_error': best_valid_error
            }
        else:
            _job_status['best'] = {}

        _job_status['last'] = {}
        if last_train_error is not None:
            _job_status['last']['train_error'] = last_train_error
        if last_valid_error is not None:
            _job_status['last']['valid_error'] = last_valid_error


def get_val(keys):
    if type(keys) is str:
        keys = keys.split('.')

    target = _job_status
    for key in keys:
        key = str(key)
        if not isinstance(target, dict) or not key in target:
            return None
        target = target[key]
    return target


def set_val(keys, value):
    if type(keys) is str:
        keys = keys.split('.')

    global _job_status
    if not isinstance(_job_status, dict):
        _job_status = {}

    target = _job_status
    for key in keys[:-1]:
        key = str(key)
        if not key in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    key = str(keys[-1])
    target[key] = value

    if key == 'status' and value in ['finished', 'failed']:
        current = utcnow_timestamp()
        set_val('end_time', current)
        if value == 'finished':
            start_time = get_val('start_time')
            elapsed = current - start_time
            set_val('time.prediction', elapsed)


def start_process(start_time=None):
    set_val('_process_start_time',
            start_time if start_time else utcnow_timestamp())


def update_time_train(prediction=None):
    start_time = get_val('start_time')
    process_start_time = get_val('_process_start_time')
    if not process_start_time:
        return

    if not prediction:
        epoch_current = get_val('epoch.current')
        epoch_max = get_val('epoch.max')
        if not epoch_max:
            return  # may be not reached
        current_time = utcnow_timestamp()
        process_elapsed = current_time - process_start_time
        prediction = process_elapsed / (epoch_current / epoch_max)

    pre_elapsed = process_start_time - start_time
    set_val('time.prediction', prediction + pre_elapsed)


def update_time_forward():
    current_time = utcnow_timestamp()
    start_time = get_val('start_time')
    process_start_time = get_val('_process_start_time')
    if not process_start_time:
        return
    epoch_current = get_val('data.current')
    epoch_max = get_val('data.max')
    if not epoch_max:
        return

    pre_elapsed = process_start_time - start_time
    process_elapsed = current_time - process_start_time
    prediction = process_elapsed / (epoch_current / epoch_max)

    set_val('time.prediction', prediction + pre_elapsed)


def update_result_forward(output_result_filename, header, rows):
    with open(os.path.join(_outdir, output_result_filename), 'r') as f:
        csv_header = f.readline().rstrip('\r\n')
    set_val('output_result.csv_header', csv_header)
    set_val('output_result.column_num', len(header))
    set_val('output_result.data_num', len(rows))


def dump(status=None):
    if status:
        set_val('status', status)

    random_str = ''.join(
        [random.choice(string.ascii_letters + string.digits) for i in range(random_num)])
    with open(os.path.join(_outdir, random_str + '_status.json'), 'w') as f:
        _job_status['update_timestamp'] = utcnow_timestamp()
        f.write(json.dumps(
            {
                k: _job_status[k]
                for k in filter(lambda k: k[:1] != '_',
                                _job_status.keys())
            },
            sort_keys=True,
            indent=4,
        ))
    shutil.move(_outdir + '/' + random_str + '_status.json',
                _outdir + '/' + 'status.json')


_snapshot_seq = 0


def save_train_snapshot():
    tmp_dir = os.path.join(_outdir, '_tmp_')
    try:
        os.mkdir(tmp_dir)
    except OSError:
        pass  # python2 does not support exists_ok arg

    nnp_files = glob.glob(os.path.join(_outdir, '*.nnp'))
    for nnp_file in nnp_files:
        shutil.copy2(nnp_file, tmp_dir)
        shutil.copy2(nnp_file, os.path.join(tmp_dir, 'result.nnp'))

    for result_file in [
            os.path.join(_outdir, 'monitoring_report.yml'),
    ]:
        try:
            shutil.copy2(result_file, tmp_dir)
        except FileNotFoundError:
            pass

    global _snapshot_seq
    _snapshot_seq += 1
    snapshot_dir = os.path.join(_outdir, '{}_snapshot'.format(_snapshot_seq))
    while os.path.exists(snapshot_dir):
        _snapshot_seq += 1
        snapshot_dir = os.path.join(
            _outdir, '{}_snapshot'.format(_snapshot_seq))

    os.rename(tmp_dir, snapshot_dir)
