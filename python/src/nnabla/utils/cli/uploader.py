import argparse
import boto3
import configparser
import csv
import math
import os
import re
import requests
import shutil
import sys
import tarfile
import tempfile
import threading
import tqdm


class Uploader:
    environments = {
        'dev': 'https://dev-api.cdd-sdeep.com',
        'stg': 'https://stg-api.cdd-sdeep.net',
        'qa': 'https://qa-api.cdd-sdeep.net',
        'prod': 'https://api.dl.sony.com/console'}

    def createCsvData(self, csvfilename):
        data_files = {}
        csv_data = []
        num_of_data_files = 0

        if os.path.exists(csvfilename):
            with open(csvfilename, 'r') as f:
                csv_lines = f.readlines()

                linecount = 0

                num_of_items = -1
                csvreader = csv.reader(csv_lines)

                self._progress.init(len(csv_lines), 'Create CSV')

                for row in csvreader:

                    self._progress(1.0)

                    # Validate
                    if linecount == 0:  # Header
                        num_of_items = len(row)
                    else:  # Data
                        if num_of_items != len(row):
                            self._log(
                                'invalid line {} in csv file.'.format(linecount))
                            return None, None

                    new_row = []
                    for item in row:
                        data_file = None
                        if os.path.isfile(item):
                            data_file = item
                        else:
                            fn = os.path.join(os.path.dirname(
                                csvfilename), os.path.join(*item.split('\\')))
                            if os.path.isfile(fn):
                                data_file = fn

                        if data_file is not None:
                            name = os.path.join('data', '{:012d}_{}'.format(
                                num_of_data_files, os.path.basename(data_file)))
                            data_files[name] = os.path.abspath(data_file)
                            new_row.append(name)
                            num_of_data_files += 1
                        else:
                            new_row.append(item)
                    csv_data.append(new_row)
                    linecount += 1
                self._progress.finish()
        return csv_data, data_files

    def createTemporaryTar(self, name, csv_data, data_files, tmpdir):
        indexcsvfilename = os.path.join(tmpdir, 'index.csv')

        self._log('Create index.csv')
        self._progress.init(len(csv_data), 'Create index.csv')
        with open(indexcsvfilename, 'w') as f:
            for row in csv_data:
                self._progress(1.0)
                f.write(','.join(row) + '\n')
        self._progress.finish()

        tarfilename = os.path.join(tmpdir, '{}.tar'.format(name))

        self._log('Add file to tar archive.')
        self._progress.init(len(data_files), 'Create TAR')
        with tarfile.open(tarfilename, 'w') as tar:
            tar.add(indexcsvfilename, 'index.csv')
            for datafile in sorted(data_files.keys()):
                self._progress(1.0)
                tar.add(data_files[datafile], datafile)
        self._progress.finish()

        return tarfilename

    def uploadFile(self, env, token, filename, name):
        size = os.path.getsize(filename)
        api_url = self.environments[env]
        self._log('DEBUG: Getting Upload path from [{}]'.format(
            api_url))  # TODO Remove

        r = requests.get('{}/v1/misc/credential'.format(api_url),
                         params={
            'encrypted_text': token,
            'dataset_name': name,
            'dataset_size': size})
        info = r.json()

        if 'upload_path' not in info:
            self._log('Server returns [{}]'.format(info['message']))
            return False

        upload_url = info['upload_path']
        self._log('DEBUG: upload_url is [{}]'.format(
            upload_url))  # TODO Remove

        bucketname, key = upload_url.split('://', 1)[1].split('/', 1)
        upload_key = '{}/{}.tar'.format(key, name)

        s3 = boto3.session.Session(aws_access_key_id=info['access_key_id'],
                                   aws_secret_access_key=info['secret_access_key'],
                                   aws_session_token=info['session_token']).client('s3')
        tc = boto3.s3.transfer.TransferConfig(
            multipart_threshold=10 * 1024 * 1024,
            max_concurrency=10)
        t = boto3.s3.transfer.S3Transfer(client=s3, config=tc)

        self._progress.init(os.path.getsize(filename), 'Upload')
        t.upload_file(filename, bucketname, upload_key,
                      callback=self._progress)
        self._progress.finish()

        return True

    def __init__(self,
                 log,
                 token=None,
                 progress=None):
        self._log = log
        self._progress = progress

    def convert(self, source, destination):
        tmpdir = tempfile.mkdtemp()
        self._log('Temprary dir {} created'.format(tmpdir))

        self._log('Prepare csv data')
        csv_data, data_files = self.createCsvData(source)
        if csv_data is not None:
            self._log('Prepare tar file')
            name = os.path.splitext(os.path.basename(source))[0]
            tarfile = self.createTemporaryTar(name,
                                              csv_data,
                                              data_files,
                                              tmpdir)
            shutil.copyfile(tarfile, destination)
        shutil.rmtree(tmpdir, ignore_errors=True)
        self._log('Temprary dir removed')

    def upload(self, token, filename, name, finishCallback=None, env='stg'):

        _, ext = os.path.splitext(filename)
        if ext == '.csv':
            tmpdir = tempfile.mkdtemp()
            self._log('Temprary dir {} created'.format(tmpdir))

            self._log('Prepare csv data')
            csv_data, data_files = self.createCsvData(filename)
            self._log('Prepare tar file')
            tarfile = self.createTemporaryTar(name,
                                              csv_data,
                                              data_files,
                                              tmpdir)
            self._log('Upload')
            res = self.uploadFile(env, token, tarfile, name)

            shutil.rmtree(tmpdir, ignore_errors=True)
            self._log('Temprary dir removed')
            if res:
                self._log('Finished')

        elif ext == '.tar':
            if self.uploadFile(env, token, filename, name):
                self._log('Finished')
        else:
            self._log('filename with extension {} is not supported.'.format(ext))
            if finishCallback is not None:
                finishCallback(False)
            return

        if finishCallback is not None:
            finishCallback(True)


class Progress:
    def __init__(self):
        self._lock = threading.Lock()
        self._pbar = None

    def __call__(self, amount):
        with self._lock:
            if self._pbar is not None:
                self._pbar.update(amount)

    def init(self, maximum, label):
        if self._pbar is not None:
            self._pbar.close()
        self._pbar = tqdm.tqdm(total=maximum, desc=label)

    def finish(self):
        self._pbar.close()


def log(string):
    print(string)


def upload_command(args):
    if not args.env:
        args.env = os.getenv("NNC_ENV", 'stg')
    if args.env not in Uploader.environments:
        print('-e option ({}) must be one of ({})'.format(args.env,
                                                          ', '.join(sorted(Uploader.environments.keys()))))
        sys.exit(-1)

    uploader = Uploader(log=log, progress=Progress())
    name = os.path.splitext(os.path.basename(args.filename))[0]
    uploader.upload(args.token, args.filename, name,
                    env=args.env)


def create_tar_command(args):
    Uploader(log=log, progress=Progress()).convert(
        args.source, args.destination)
