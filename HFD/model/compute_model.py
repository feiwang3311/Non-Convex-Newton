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
  def __init__(self, model):
    super(myMLP, self).__init__()
  # Evaluting neural network models: loss, grad, Hess(), error
  # Input:
  #       model       ---- specification neural network model
  #         model.numlayers: number of layers
  #         model.layersizes: number of neurons of each layer
  #         model.layertypes: type of each layer (currently only support
  #                           logisitic(sigmoid),tanh,linear,softmax)
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
    self.model = model
    self.numLayers = model.numlayers
    self.layerTypes = model.layertypes
    self.layerSizes = model.layersizes
    self.lossType = model.layertypes[model.numlayers -1]
    layers = []
    self.params = []
    for i in range(model.numlayers):
      layer = nn.Linear(model.layersizes[i], model.layersizes[i+1])
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

  def params(self):
    # this function returns a (flatten??) params (all weights and biases)
    return self.params

  def paramsNorm(self, norm_id):
    params = [p.detach().numpy() for p in self.params]
    param_flatten = np.concatenate([p.reshape(-1) for p in params])
    return np.linalg.norm(param_flatten, norm_id)

  def parameterSize(self):
    psize = [(self.layerSizes[i]+1) * self.layerSizes[i+1] for i in range(self.numLayers)]
    psize = sum(psize)
    return psize

  def updateParameters(self, updates):
    for (param, update) in zip(self.params, updates):
      param.data.add_(update) # update will be numpy array

  def setupParameters(self, parameters):
    assert(False)

  def loss(self, X, y):
    logits = self.layers(X)
    if self.lossType == 'linear':
      loss = nn.MSELoss()
      return loss(logits, y), 0.0 # TODO: compute and return the accuracy?
    elif self.lossType == 'softmax':
      logits = F.log_softmax(logits)
      loss = nn.NLLLoss()
      return loss(logits, y), 0.0

  def gradient(self, X, y):
    loss, accu = self.loss(X, y)
    grads = torch.autograd.grad(loss, self.params, create_graph=True, retain_graph=True)
    flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
    return (loss, accu), flatten

  def hess(self, X, y):
    def hessVec(vc):
      ((_,_), flatten) = self.gradient(X, y)
      hvps = torch.autograd.grad([flatten @ torch.from_numpy(vc.astype(np.float32))], self.params, allow_unused=True)
      h_flatten = torch.cat([h.reshape(-1) for h in hvps if h is not None])
      return h_flatten.numpy()
    return hessVec
