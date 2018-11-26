import csv
import glob
import numpy as np
import os
import tempfile
import zipfile
from shutil import rmtree

from nnabla.logger import logger
from nnabla.parameter import get_parameter_or_create
from nnabla.parameter import load_parameters

# ##############################################################################
# Train


def _get_best_param(paramlist):
    h5list = []
    bestlist = {}
    currentlist = {}
    # with newest spec best param store as 'results.nnp'
    if 'results.nnp' in paramlist:
        return 'results.nnp'
    for fn in paramlist:
        name, ext = os.path.splitext(fn)
        if ext == '.h5':
            h5.append(ext)
        elif ext == '.nnp':
            ns = name.split('_')
            if len(ns) == 3:
                if ns[0] == 'results':
                    if ns[1] == 'best':
                        bestlist[int(ns[2])] = fn
                    elif ns[1] == 'current':
                        currentlist[int(ns[2])] = fn
    if len(bestlist) > 0:
        return bestlist[sorted(bestlist.keys()).pop()]
    elif len(currentlist) > 0:
        return currentlist[sorted(currentlist.keys()).pop()]
    elif len(h5list) > 0:
        return sorted(h5list).pop()
    return None


def get_info_from_sdcproj(args):
    if args.sdcproj and args.job_url_list:
        job_url_list = {}
        with open(args.job_url_list) as f:
            for line in f.readlines():
                ls = line.strip().split()
                if len(ls) == 2:
                    job_url_list[ls[0]] = ls[1]

        param_list = {}
        if args.assign is not None:
            with open(args.assign) as f:
                for line in f.readlines():
                    ls = line.strip().split(',')
                    if len(ls) == 2:
                        src, dst = ls
                        src = src.strip()
                        dst = dst.strip()
                        job_id, param = src.split('/', 1)
                        if job_id in job_url_list:
                            uri = job_url_list[job_id]
                            if uri not in param_list:
                                param_list[uri] = {}
                            if param in param_list[uri] and param_list[uri][param] != dst:
                                logger.log(99, '{} in {} duplicated between {} and {}'.format(
                                    param, uri, dst, param_list[uri][param]))
                                sys.exit(-1)
                            param_list[uri][param] = dst

        for uri, params in param_list.items():
            param_proto = None
            param_fn = None
            if uri[0:5].lower() == 's3://':
                uri_header, uri_body = uri.split('://', 1)
                us = uri_body.split('/', 1)
                bucketname = us.pop(0)
                base_key = us[0]
                logger.info(
                    'Creating session for S3 bucket {}'.format(bucketname))
                import boto3
                bucket = boto3.session.Session().resource('s3').Bucket(bucketname)
                paramlist = []
                for obj in bucket.objects.filter(Prefix=base_key):
                    fn = obj.key[len(base_key) + 1:]
                    if len(fn) > 0:
                        paramlist.append(fn)
                p = _get_best_param(paramlist)
                if p is not None:
                    param_fn = uri + '/' + p
                    tempdir = tempfile.mkdtemp()
                    tmp = os.path.join(tempdir, p)
                    with open(tmp, 'wb') as f:
                        f.write(bucket.Object(
                            base_key + '/' + p).get()['Body'].read())
                    param_proto = load_parameters(tmp, proto_only=True)
                    rmtree(tempdir, ignore_errors=True)

            else:
                paramlist = []
                for fn in glob.glob('{}/*'.format(uri)):
                    paramlist.append(os.path.basename(fn))
                p = _get_best_param(paramlist)
                if p is not None:
                    param_fn = os.path.join(uri, p)
                    param_proto = load_parameters(param_fn, proto_only=True)

            if param_proto is not None:
                for param in param_proto.parameter:
                    pn = param.variable_name.replace('/', '~')
                    if pn in params:
                        dst = params[pn]
                        logger.log(99, 'Update variable {} from {}({})'.format(
                            dst, param_fn, pn))
                        var = get_parameter_or_create(dst, param.shape.dim)
                        var.d = np.reshape(param.data, param.shape.dim)
                        var.need_grad = param.need_grad

    timelimit = -1
    if args.sdcproj:
        with open(args.sdcproj) as f:
            for line in f.readlines():
                ls = line.strip().split('=')
                if len(ls) == 2:
                    var, val = ls
                    if var == 'TimeLimit' and val:
                        timelimits = [int(x) for x in val.split(':')]
                        if len(timelimits) == 4:
                            timelimit = float(timelimits[0] * 24 * 3600 +
                                              timelimits[1] * 3600 +
                                              timelimits[2] * 60 + timelimits[3])
    return timelimit

# ##############################################################################
# Forward


def add_evaluation_result_to_nnp(args, row0, rows):
    images = {}
    ref_image_num = 0
    new_rows = []
    for row in rows:
        new_row = []
        for item in row:

            isfile = False
            try:
                isfile = os.path.isfile(item)
            except:
                isfile = False

            if isfile:
                if os.path.isabs(item):
                    image_filename = item
                    image_name = './reference/{}_{}'.format(
                        ref_image_num, os.path.basename(item))
                    ref_image_num += 1
                    item = image_name
                    images[image_name] = image_filename
                else:
                    image_filename = item
                    image_name = item
                    images[image_name] = image_filename
            else:

                isfile = False
                try:
                    isfile = os.path.isfile(os.path.join(args.outdir, item))
                except:
                    isfile = False

                if isfile:
                    image_filename = os.path.join(args.outdir, item)
                    image_name = item
                    images[image_name] = image_filename

            new_row.append(item)
        new_rows.append(new_row)

    result_csv_filename_tmp = os.path.join(
        args.outdir, 'output_result_tmp.csv')

    with open(result_csv_filename_tmp, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(row0)
        writer.writerows(new_rows)

    result_zip_filename = os.path.join(args.outdir, 'output_result.zip')
    with zipfile.ZipFile(result_zip_filename, 'w') as res:
        res.write(result_csv_filename_tmp, 'output_result.csv')
        for name, filename in images.items():
            res.write(filename, name)
    globname = os.path.join(args.outdir, 'result*.nnp')
    exists = glob.glob(globname)
    last_results = []
    if len(exists) > 0:
        last_result_nums = {}
        for ex in exists:
            name = os.path.basename(ex).rsplit('.', 1)[0]
            try:
                name_base, num = name.rsplit('_', 1)
                num = int(num)
                if name_base not in last_result_nums:
                    last_result_nums[name_base] = []
                last_result_nums[name_base].append(num)
            except:
                last_results.append(os.path.basename(ex))
        for base in last_result_nums.keys():
            last_results.append('{}_{}.nnp'.format(
                base, sorted(last_result_nums[base]).pop()))

    for result_filename in last_results:
        result_zip_name = 'output_result.zip'
        result_zip_num = 1
        with zipfile.ZipFile(os.path.join(args.outdir, result_filename), 'a') as res:
            while result_zip_name in res.namelist():
                result_zip_name = 'output_result_{}.zip'.format(result_zip_num)
                result_zip_num += 1
            logger.log(99, 'Add {} to {}.'.format(
                result_zip_name, result_filename))
            res.write(result_zip_filename, result_zip_name)
