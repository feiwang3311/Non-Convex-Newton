from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pickle
import collections
from sklearn import preprocessing as prep
from algorithms.subsampled_tr_cg import subsampled_tr_cg

# notes about cifar10 data:
# there are 5 batches for training and 1 batch for testing
# each batch is a pickle file, that is a dict with 2 field:
# the data field is a 10000 x 3072 numpy array of uint8, which is 10000 input images
# the lables field is a list of 10000 numbers in the range of 0-9, which is the label
def unpickle(f):
  with open(f, 'rb') as fo:
    dict = pickle.load(fo, encoding = 'bytes')
  return dict

def readCifar10Data():
    training_files = 'cifar10_data/cifar-10-batches-py/data_batch_1' # TODO: this is just one batch, there are 5 total
    training_dict = unpickle(training_files)
    testing_files = 'cifar10_data/cifar-10-batches-py/test_batch' # there is only one batch for testing
    testing_dict = unpickle(testing_files)
    return training_dict[b'data'], training_dict[b'labels'], testing_dict[b'data'], testing_dict[b'labels']

def normalizeData(data):
    return prep.normalize(data).astype(np.float32)

def normalize(X, y, X_test, y_test):
    datas = [normalizeData(X), np.array(y), normalizeData(X_test), np.array(y_test)]
    return [Variable(torch.from_numpy(data)) for data in datas]

def cifar_classification():
    parser = argparse.ArgumentParser(description='SecondOrderOptimization cifar11 example')
    parser.add_argument('--method', type=str, default='TR-CG', help='method for optimization, pick from TR-CG, GN, SGD')
    parser.add_argument('--hs_sub', type=float, default=0.05, help='sampling ratio of the training set')
    parser.add_argument('--init',   type=int, default=0, help='initialization schemes: 0-zeros 1-normal 2-random')
    parser.add_argument('--delta',  type=float, default = 1000, help='initial trust-region radius for TR methods')
    parser.add_argument('--alpha',  type=float, default = 0.05, help='step size for SGD')
    parser.add_argument('--maxNP',  type=int, default = 1e8, help='maximum propagations')
    parser.add_argument('--seed',   type=int, default = 0, help='random seed')

    # cifar10 classification using 1-hidden layer network.
    #
    # Input:
    #       method      ---- TR-CG, GN, SGD
    #       hs_sub      ---- sampling ratio of the training set, e.g. 0.1
    #       init        ---- initialization schemes
    #                       0: zeros initialization
    #                       1: normalized random initialization
    #                       2: random intialization
    #       delta       ---- initial trust-region radius for TR methods
    #       alpha       ---- step size for SGD algorithm
    #       maxNP       ---- maximum propagations
    #
    # Output:
    #       options     ---- contain all the training information and results;
    #                        see each algorithm function for details.
    #
    #           options.params:     weights of the model
    #           options.tr_times:   training timestamp at each iteration
    #           options.tr_losses:  training loss at each iteration
    #           options.tr_grads:   training gradient norm at each iteration
    #           opitons.tr_errs:    training error at each iteration
    #           options.te_errs:    test error at each iteration
    #           options.cur_ter:    current itertion number(if not 0, resume
    #                               training
    #
    X, y, X_test, y_test = readCifar10Data()
    n = X.shape[0]
    inputd = X.shape[1]
    outputd = 10
    X, y, X_test, y_test = normalize(X, y, X_test, y_test)

    Model = collections.namedtuple('Model', ['layersizes','layertypes','numlayers','type'])
    layerSize = 512
    model = Model(layersizes = [inputd, layerSize, outputd],
                  layertypes = ['logistic', 'softmax'],
                  numlayers  = 2,
                  type = 'classification')
    psize = inputd * layerSize + layerSize
    lamda = 0 # l2 regularization

    options = parser.parse_args()
    options.name = 'cifar10_classification';
    options.inner_iters = 250;
    options.max_delta = float('Inf')
    options.max_iters = 1e6;
    options.cur_iter = 0;
    options.hs = math.floor(options.hs_sub*n); #  Hessian batch size for 2nd-order Methods, gradient batch size for SGD.
    options.maxMVPs = float('Inf');
    options.maxNoProps = options.maxNP;

    if options.init == 0:
      sub_dir = "".join(['/zeros_', str(options.seed)])
    elif options.init == 1:
      sub_dir = "".join(['randn_normalized_', str(options.seed)])
    else:
      sub_dir = "".join(['/randn', str(options.seed)])
    dir_name = "".join(['./results/',options.name,sub_dir]);
    if not os.path.isdir(dir_name):
      os.makedirs(dir_name)
    file_name = "".join([dir_name,'/',options.name,'_lambda_', str(lamda), '_hess_', str(options.hs_sub)])
    if options.method == 'GN':
      print('\n\n------------------- GN %g ----------------\n\n' % options.delta)
      file_name_gn = "".join([file_name, '_gn.mat'])
      if os.path.isfile(file_name_gn):
        load(file_name_gn, 'options') # resume training
      (params, options) = subsampled_gn(model,X,y,X_test,y_test,lamda,options)
      parsave(file_name_gn, options)

    elif options.method == 'TR-CG':
      print('\n\n------------------- TR: delta = %g ----------------\n\n' % options.delta)
      file_name_tr_cg = "".join([file_name,'_tr_cg','_delta_',str(options.delta),'.mat'])
      if os.path.isfile(file_name_tr_cg):
        load(file_name_tr_cg, 'options') # resume training
      (params, options) = subsampled_tr_cg(model,X,y,X_test,y_test,lamda,options)
      parsave(file_name_tr_cg, options)

    elif options.method =='SGD':
      print('\n\n------------------- SGD: alpha = %g ----------------\n\n' % options.alpha)
      file_name_sgd = "".join([file_name,'_step_', str(options.alpha), '_sgd.mat'])
      if os.path.isfile(file_name_sgd):
        load(file_name_sgd,'options') # resume training
      (params, options) = momentum_sgd(model,X,y,X_test,y_test,lamda,options);
      parsave(file_name_sgd, options)
    return options

def parsave(file_name, options):
  # save a pickle dump
  with open(file_name, 'w') as f:
    pickle.dump(options, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  cifar_classification()
