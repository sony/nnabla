# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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
import boto3
import csv
import json
import os
import urllib.parse
import urllib.request as request
import shutil
import tarfile
import tempfile
import threading
import tqdm


class Uploader:
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
                            fn = '/'.join([os.path.dirname(
                                csvfilename), '/'.join(item.split('\\'))])
                            if os.path.isfile(fn):
                                data_file = fn

                        if data_file is not None:
                            name = '/'.join(['data', '{:012d}_{}'.format(
                                num_of_data_files, os.path.basename(data_file))])
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
        with open(indexcsvfilename, 'w', newline='') as f:
            csvwriter = csv.writer(f)
            for row in csv_data:
                csvwriter.writerow(row)
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

    def uploadFile(self, endpoint, token, filename, name):
        size = os.path.getsize(filename)
        if endpoint == 'https://console-api.dl.sony.com':
            self._log('Getting Upload path')
        else:
            self._log('Getting Upload path from [{}]'.format(endpoint))

        params = urllib.parse.urlencode({
            'encrypted_text': token,
            'dataset_name': name,
            'dataset_size': size
        })
        r = request.urlopen(f'{endpoint}/v1/misc/credential?{params}')
        info = json.loads(r.read().decode())

        if 'upload_path' not in info:
            if endpoint == 'https://console-api.dl.sony.com':
                self._log('Upload_path could not be retrieved from the server.')
            else:
                self._log('Server returns [{}]'.format(info['message']))
            return False

        upload_url = info['upload_path']
        if endpoint == 'https://console-api.dl.sony.com':
            self._log('Got upload_url')
        else:
            self._log('upload_url is [{}]'.format(
                upload_url))

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
        self._log('Temporary dir {} created'.format(tmpdir))

        try:
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
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            self._log('Temporary dir removed')

    def upload(self, token, filename, name, finishCallback=None, endpoint=None):

        _, ext = os.path.splitext(filename)
        if ext == '.csv':
            tmpdir = tempfile.mkdtemp()
            self._log('Temporary dir {} created'.format(tmpdir))

            try:
                self._log('Prepare csv data')
                csv_data, data_files = self.createCsvData(filename)
                self._log('Prepare tar file')
                tarfile = self.createTemporaryTar(name,
                                                  csv_data,
                                                  data_files,
                                                  tmpdir)
                self._log('Upload')
                res = self.uploadFile(endpoint, token, tarfile, name)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
                self._log('Temporary dir removed')

            if res:
                self._log('Finished')

        elif ext == '.tar':
            if self.uploadFile(endpoint, token, filename, name):
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
    if not args.endpoint:
        args.endpoint = os.getenv(
            "NNC_ENDPOINT", 'https://console-api.dl.sony.com')

    uploader = Uploader(log=log, progress=Progress())
    name = os.path.splitext(os.path.basename(args.filename))[0]
    uploader.upload(args.token, args.filename, name,
                    endpoint=args.endpoint)
    return True


def add_upload_command(subparsers):
    # Uploader
    from nnabla.utils.cli.uploader import upload_command
    subparser = subparsers.add_parser(
        'upload', help='Upload dataset to Neural Network Console.')
    subparser.add_argument(
        '-e', '--endpoint', help='set endpoint uri', type=str)
    subparser.add_argument('token', help='token for upload')
    subparser.add_argument('filename', help='filename to upload')
    subparser.set_defaults(func=upload_command)


def create_tar_command(args):
    Uploader(log=log, progress=Progress()).convert(
        args.source, args.destination)


def add_create_tar_command(subparsers):
    # Create TAR for uploader
    from nnabla.utils.cli.uploader import create_tar_command
    subparser = subparsers.add_parser(
        'create_tar', help='Create tar file for Neural Network Console.')
    subparser.add_argument('source', help='CSV dataset')
    subparser.add_argument('destination', help='TAR filename')
    subparser.set_defaults(func=create_tar_command)
