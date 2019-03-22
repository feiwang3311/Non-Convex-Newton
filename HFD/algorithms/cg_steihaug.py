import numpy as np
import math

def cg_steihaug(H, g, delta, params, x0 = None):

  # ... This procedure approximately solves the following trust region
  #     problem
  #
  #         minimize    m(p) = 1/2 p'Hp + g'p
  #         subject to  ||p|| <= Delta
  #
  #     by means of the CG-Steihaug method.
  #
  #--------------------------------------------------------------------------
  # INPUT
  #
  #             H:  Hessian matrix
  #             g:  gradient vector
  #         delta:  radius of the TR
  #     params(1):  relative residual reduction factor
  #     params(2):  max number of iterations
  #     params(3):  level of output
  #
  # OUTPUT
  #             p:  an aproximate solution of (1)-(2)
  #        num_cg:  number of CG iterations to achieve convergence
  #         iflag:  termination condition
  #
  #--------------------------------------------------------------------------
  g = g.detach().numpy()
  tr_model = lambda x: 0.5 * np.inner(x, H(x)) + np.inner(x, g)
  n = g.size
  errtol = params[0]
  maxit  = params[1]
  iprnt  = params[2]
  iflag  = ' '

  if x0 is not None:
    x = x0
  else:
    x = np.zeros((n, 1))
  r = - g - H(x)
  z = r
  rho = np.inner(z, r)
  tst = np.linalg.norm(r, 2)
  flag = ''
  terminate = errtol * tst
  it = 1
  hatdel = delta * 1
  rhoold = 1.0 # TODO: not sure if this is the same as 1.0d0 in matlab
  if iprnt > 0:
    print('This is an output of the CG-Steihaug method. \n\tDelta = %7.1e \n' % delta)
    print('---------------------------------------------------------------\n')

  flag = 'We do not know '
  if tst <= terminate:
    flag = 'Small ||g||'

  while ((tst > terminate) and (it <= maxit) and np.linalg.norm(x) <= hatdel):
    if (it == 1):
      p = z
    else:
      beta = rho / rhoold
      p = z + beta * p

    w = H(p)
    alpha = np.inner(w, p)

    ineg = 0
    if (alpha <= 0):
      ac = np.inner(p, p)
      bc = 2 * np.inner(x, p)
      cc = np.inner(x, x) - delta * delta
      alpha = (-bc + math.sqrt(bc * bc - 4 * ac * cc)) / (2 * ac)
      flag = 'negative curvature'
      iflag = 'NC'
      x = x + alpha * p
      break
    else:
      alpha = rho / alpha
      if np.linalg.norm(x + alpha * p, 2) > delta:
        ac = np.inner(p, p)
        bc = 2 * np.inner(x, p)
        cc = np.inner(x, x) - delta * delta
        alpha = (-bc + math.sqrt(bc * bc - 4 * ac * cc)) / (2 * ac)
        flag = 'boundary was hit'
        iflag = 'TR'
        x = x + alpha * p
        break

    x = x + alpha * p
    r = r - alpha * w
    tst = np.linalg.norm(r, 2)
    if tst <= terminate:
      flag = '||r|| < test'
      iflag = 'RS'
      break
    if np.linalg.norm(x) >= hatdel:
      flag = 'close to the boundary'
      iflag = 'TR'
      break

    if iprnt > 0:
      print(' %3i     %14.8e      %s    \n', it, tst, flag)
    rhoold = rho
    z = r
    rho = np.inner(z, r)
    it = it + 1

  if it > maxit:
    iflag = 'MX'

  num_cg = it
  p = x
  m = tr_model(p)
  return p, m, num_cg, iflag
