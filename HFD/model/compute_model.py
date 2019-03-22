from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class myMLP(nn.Module):
  def __init__(self, model, init = None):
    super(myMLP, self).__init__()
  # Evaluting neural network models: loss, grad, Hess(), error
  # Input:
  #       model       ---- specification neural network model
  #         model.numlayers: number of layers
  #         model.layersizes: number of neurons of each layer
  #         model.layertypes: type of each layer (currently only support
  #                           logisitic(sigmoid),tanh,linear,softmax)
  #       init        ---- specification of how initialization happens
  #         0: all initialization are done as zero
  #         1: all initialization are done by random normal
  #         2: all initialization are done by random uniform
  #
  # Output:
  #       loss        ---- loss
  #       grad        ---- gradient, same size as params
  #       hess        ---- a function compute Hessian-vector product.
  #       perr        ---- learning error,e.g. classification error or mse.
  #    if only returning 1 output, it returns [loss, learning error]
  #     i.e. res = compute_model(model,params,X,y) --> res = [loss, err]
  #    if returning 2 outputs,  it returns [[loss,err], gradient]
  #     i.e. [res,grad] = compute_model(model,params,X,y) -> res = [loss, err]
  #    if returning 3 outputs, it returns [loss, grad, hess]
    self.init = init
    self.model = model
    self.numLayers = model.numlayers
    self.layerTypes = model.layertypes
    self.layerSizes = model.layersizes
    self.lossType = model.layertypes[model.numlayers -1]
    layers = []
    self.params = []
    for i in range(model.numlayers):
      layer = nn.Linear(model.layersizes[i], model.layersizes[i+1])
      # TODO: based on init value, reset the parameters of layer
      # if init == 0: layer.apply(weight_init_0)
      # if init == 1: layer.apply(weight_init_1)
      # if init == 2: layer.apply(weight_init_2)
      layers.append(('linear_{}'.format(i), layer))
      self.params.append(layer.weight)
      self.params.append(layer.bias) # TODO: arrange the weight and bias in this way?
      if model.layertypes[i] == 'logistic':
        layers.append(('logistic_{}'.format(i), nn.Sigmoid()))
      elif model.layertypes[i] == 'tanh':
        layers.append(('tanh_{}'.format(i), nn.Tanh()))
      elif model.layertypes[i] == 'linear':
        pass
      elif model.layertypes[i] == 'softmax':
        layers.append(('softmax_{}'.format(i), nn.Softmax()))
      else:
        assert False, '{} activation layer is not implemented'.format(model.layertypes[i])
    self.layers = nn.Sequential(collections.OrderedDict(layers))
    psize = [(self.layerSizes[i]+1) * self.layerSizes[i+1] for i in range(self.numLayers)]
    self.psize = sum(psize)

  def flatten_params(self):
    # this function returns a (flatten) params (all weights and biases)
    # TODO: cache the flatten parameters??
    params = [p.detach().numpy() for p in self.params]
    param_flatten = np.concatenate([p.reshape(-1) for p in params])
    return param_flatten

  # deprecated
  def paramsNorm(self, norm_id):
    return np.linalg.norm(self.flatten_params(), norm_id)

  def parameterSize(self):
    return self.psize

  def updateParameters(self, updates):
    for (param, update) in zip(self.params, updates):
      param.data.add_(update) # update will be numpy array

  def setupParameters(self, parameters):
    assert False, "not implemented"

  def loss(self, X, y):
    logits = self.layers(X)
    if self.lossType == 'linear':
      loss = nn.MSELoss()
      return loss(logits, y), 0.0 # TODO: compute and return the accuracy?
    elif self.lossType == 'softmax':
      logits = F.log_softmax(logits)
      loss = nn.NLLLoss()
      return loss(logits, y), 0.0

  # private function
  def get_grad(self, X, y):
    # TODO: cache the result?
    loss, accu = self.loss(X, y)
    grads = torch.autograd.grad(loss, self.params, create_graph=True, retain_graph=True)
    return loss, accu, grads

  def gradient(self, X, y):
    loss, accu, grads = self.get_grad(X, y)
    return (loss, accu), torch2flatten_numpy(grads)

  def hess(self, X, y):
    def hessVec(vc):
      # TODO: cannot lift this out of hessVec function (it breaks pytorch backpropagation)
      # infact, this is probably also the case in Lantern. We cannot reuse the computation for gradient to compute Hvp for a different vector
      _, _, grads = self.get_grad(X, y)
      flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
      hvps = torch.autograd.grad([flatten @ torch.from_numpy(vc.astype(np.float32))], self.params, allow_unused=True)
      return torch2flatten_numpy(hvps)
    return hessVec

def torch2flatten_numpy(torches):
  return np.concatenate([t.detach().numpy().reshape(-1) for t in torches])

# this function init weight/bias to zero
# this is how you use it:
# a = nn.Linear(2, 3)
# a.apply(weight_init_0)
def weight_init_0(m):
  m.weigth.data.zero_()
  m.bias.data.zero_()

def weight_init_1(m): # this function init weight/bias to random norm
  assert False, "not yet implemented"

def weight_init_2(m): # this function init weight/bias to random uniform
  assert false, "not yet implemented"
