from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import argparse
import numpy as np
from model.compute_model import myMLP
from algorithms.cg_steihaug import cg_steihaug

def subsampled_tr_cg(model, X, y, X_test, y_test, lamda, options):
  # subsampled trust-region method for deep learning
  # input & ouput:
  #       model           ---- neural network model
  #       X,y             ---- training data: input (d x n), output (c x n)
  #       X_test,y_test   ---- test data
  #       lamda           ---- l2 regularization
  # .     options:
  #           options.params:     weights of the model
  #           options.tr_times:   training timestamp at each iteration
  #           options.tr_losses:  training loss at each iteration
  #           options.tr_grads:   training gradient norm at each iteration
  #           opitons.tr_errs:    training error at each iteration
  #           options.te_errs:    test error at each iteration
  #           options.cur_ter:    current itertion number(if not 0, resume
  #                               training
  #           options.maxNoProps: maximum propagations for training
  #           options.max_iters:  maximum iterations for training
  #
  #           options.name:       trust-region
  #           options.delta:      initial trust-region radius
  #           options.eta1,eta2,gamma1,gamma2:
  #                               parameters for adaptivity
  #           options.hs:         subsampling Hessian size

  net = myMLP(model, options.init) # how are parameter initialized
  psize = net.parameterSize()
  layersizes = model.layersizes
  numlayers = model.numlayers
  noProps = 1
  noMVPs = 1
  n = X.shape[0]
  sz = math.floor(0.05 * n)

  delta = 5
  max_delta = 20
  eta1 = 0.8
  eta2 = 0.0001
  gamma1 = 2
  gamma2 = 1.2
  maxNoProps = float('Inf')
  maxMVPs = float('Inf')
  max_iters = 100
  inner_iters = 100
  cur = 0
  cur_time = 0
  name = 'trust-region'

  if hasattr('options', 'delta'):
      delta = options.delta
  if hasattr('options', 'maxNoProps'):
      maxNoProps = options.maxNoProps
  if hasattr('options', 'maxMVPs'):
      maxMVPs = options.maxMVPs
  if hasattr('options', 'max_delta'):
      max_delta = options.max_delta
  if hasattr('options', 'eta1'):
      eta1 = options.eta1
      gamma1 = options.gamma1
  if hasattr('options','eta2'):
      eta2 = options.eta2
      gamma2 = options.gamma2
  if hasattr('options','hs'):
      sz = options.hs
  if hasattr('options','max_iters'):
      max_iters = options.max_iters
  if hasattr('options','inner_iters'):
      inner_iters = options.inner_iters
  if hasattr('options','cur_iter') and options.cur_iter >= 1:
    cur = options.cur_iter
    cur_time = options.tr_times(cur)
    options.tr_times = np.concatenate((options.tr_times[:cur], np.zeros(max_iters)))
    options.tr_losses = np.concatenate((options.tr_losses[:cur], np.zeros(max_iters)))
    options.tr_grad = np.concatenate((options.tr_grad[:cur], np.zeros(max_iters)))
    options.tr_errs = np.concatenate((options.tr_errs[:cur], np.zeros(max_iters)))
    options.tr_noProps = np.concatenate((options.tr_noProps[:cur], np.zeros(max_iters)))
    options.tr_noMVPs = np.concatenate((options.tr_noMVPs[:cur], np.zeros(max_iters)))

    options.te_losses = np.concatenate((options.te_losses[:cur], np.zeros(max_iters)))
    options.te_errs = np.concatenate((options.te_errs[:cur], np.zeros(max_iters)))

    noProps = options.tr_noProps(cur);
    noMVPs = options.tr_noMVPs(cur);
    maxMVPs = maxMVPs + noMVPs;
  else:
    options.tr_errs = np.zeros(max_iters)
    options.tr_losses = np.zeros(max_iters)
    options.tr_grad = np.zeros(max_iters)
    options.tr_times = np.zeros(max_iters)
    options.tr_noProps = np.zeros(max_iters)
    options.tr_noMVPs = np.zeros(max_iters)

    options.te_errs = np.zeros(max_iters)
    options.te_losses = np.zeros(max_iters)

  if hasattr('options','name'):
      name = options.name
  print('initial setup of parameters:')
  if hasattr('options', 'params'):
      model.setupParameters(option.params)
  param = None # NOTE: param will be part of the model

  # initialize parameters
  print('initial setup:');
  print(' hession size: %d\n eta1: %g\n eta2: %g\n gamma1: %g\n gamma2: %g\n' % (sz, eta1, eta2, gamma1, gamma2))
  print(' init delta: %g\n max delta : %g\n max iters for solver: %d\n max props: %g\n\n' % (delta, max_delta, inner_iters, maxNoProps))

  # add tic:
  tic = time.time()
  # training
  print('\n start training...\n')
  for iter in range(cur, cur + max_iters):
    if noProps > maxNoProps or noMVPs > maxMVPs:
      break;
    idx = np.random.choice(n, sz)
    x_sample = X[idx]
    y_sample = y[idx]
    # hess is a function v => hessian dot v
    hess = net.hess(x_sample, y_sample) # (_, _, hess, _) = compute_model(model, params, x_sample, y_sample)
    (ll_err, grad) = net.gradient(X, y) # (ll_err, grad)  = compute_model(model, params, X, y)
    ll = ll_err[0]
    tr_err = ll_err[1]

    # this needs to be done for every loop because params is updated (TODO: double check)
    params = net.flatten_params()

    tr_loss = ll + 0.5 * lamda * np.inner(params, params)
    grad = grad + lamda * params
    HessV = lambda v: hess(v) + lamda * v

    noProps = noProps + X.shape[0]  # size(X,2)
    te_loss_err = net.loss(X_test, y_test);
    te_loss = te_loss_err[0]
    te_err = te_loss_err[1]

    options.tr_losses[iter] = tr_loss
    options.tr_errs[iter] = tr_err
    options.te_losses[iter] = te_loss
    options.te_errs[iter] = te_err
    grad_norm_inf = np.linalg.norm(grad, np.inf)
    grad_norm_2   = np.linalg.norm(grad, 2)
    options.tr_grad[iter] = grad_norm_inf
    options.tr_noProps[iter] = noProps
    options.tr_noMVPs[iter] = noMVPs
    options.tr_times[iter] = time.time() - tic + cur_time;
    print('\nIter: %d, time = %g s\n' % (iter, options.tr_times[iter]))
    print('training loss + reg: %g, grad: %g(max), %g(norm)\n' % (tr_loss, grad_norm_inf, grad_norm_2))
    print('training err: %g\n' % tr_err)
    print('test loss: %g, test err: %g\n' % (te_loss, te_err))
    if grad_norm_inf <= 1E-16: # 1E-6
      print('Grad too small: %g\n' % grad_norm_inf)
      break

    # solve trust-region subproblem
    fail_count = 0;
    while True:
      steihaugParams = [1e-9, 250, 0];
      if fail_count == 0:
          s0 = np.random.normal(size = psize)
          # s0 = randn(psize,1);
          s0 = 0.99*delta*s0/np.linalg.norm(s0, 2)
      (s, m, num_cg, iflag) = cg_steihaug(HessV, grad, delta, steihaugParams, s0 )
      print('Steihaug solution: %s\n' % iflag)
      noProps = noProps + num_cg * 2 * x_sample.shape[0]
      noMVPs = noMVPs + num_cg
      if m >= 0:
          s = 0
          break
      print('model reduction: %g\n' % m)
      net.updateParameters(s)
      newll_err = net.loss(X,y)
      newll = newll_err[0]
      noProps = noProps + X.shape[0]
      newll = newll + 0.5 * lamda * net.paramsNorm(2)**2
      rho = (tr_loss - newll)/-m
      if rho < eta2:
          fail_count = fail_count + 1;
          print('FALIURE No. %d: delta = %g, rho = %g, iters: %g\n' % (fail_count, delta, rho,num_cg))
          delta = delta/gamma1
          s0 = delta*s/np.linalg.norm(s, 2)
          net.updateParameters(-s) # if fail, we have to revert the parameter update
      elif rho < eta1:
          print('SUCCESS: delta = %g, rho = %g, s = %g\niters: %g, total MVPs: %g, total Props: %g\n' % (delta, rho, np.linalg.norm(s, 2), num_cg, noMVPs, noProps))
          # params = params + s # already updated before the branches
          delta = min(max_delta, gamma2*delta)
          break
      else:
          print('SUPER SUCCESS: delta = %g, rho = %g, s = %g\niters: %g, total MVPs: %g, total Props: %g\n' % (delta, rho, np.linalg.norm(s,2),num_cg,noMVPs, noProps))
          # params = params + s # already updated before the branches
          delta = min(max_delta, gamma1*delta)
          break;

  options.params = params;
  options.cur_iter = iter;
  options.tr_times = options.tr_times[:iter]
  options.tr_losses = options.tr_losses[:iter]
  options.tr_errs = options.tr_errs[:iter]
  options.tr_grad = options.tr_grad[:iter]
  options.te_losses = options.te_losses[:iter]
  options.te_errs = options.te_errs[:iter]
  options.tr_noProps = options.tr_noProps[:iter]
  options.tr_noMVPs = options.tr_noMVPs[:iter]
  return params, options
