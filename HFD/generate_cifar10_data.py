# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Read CIFAR-10 data from pickled numpy arrays and writes TFRecords.
Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR-10 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import urllib.request
import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_FILENAME_BINARY = 'cifar-10-binary.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_DOWNLOAD_URL_BINARY = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME_BINARY
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'

def maybe_download(filename, work_directory, download_url):
  """Download the data unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(download_url, filepath)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath

def download_and_extract(data_dir):
  # download CIFAR-10 if not already downloaded.
  maybe_download(CIFAR_FILENAME, data_dir, CIFAR_DOWNLOAD_URL)
  maybe_download(CIFAR_FILENAME_BINARY, data_dir, CIFAR_DOWNLOAD_URL_BINARY)
  tarfile.open(os.path.join(data_dir, CIFAR_FILENAME),
               'r:gz').extractall(data_dir)
  tarfile.open(os.path.join(data_dir, CIFAR_FILENAME_BINARY),
               'r:gz').extractall(data_dir)


def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  file_names['train'] = ['data_batch_1']
  file_names['validation'] = ['data_batch_5']
  file_names['eval'] = ['test_batch']
  return file_names


def main(data_dir):
  print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
  download_and_extract(data_dir)
  print('Done!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      default='',
      help='Directory to download and extract CIFAR-10 to.')

  args = parser.parse_args()
  main(args.data_dir)
