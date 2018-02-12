# ported from AMP.M
import numpy as np
from numpy import dot
from numpy import sqrt
from numpy import abs
from numpy import sign


def eta(data, th):
    """
    threshold function!
    """
    I = abs(data)>th
    val = (abs(data) - th)*sign(data)

    return I*val

def dEta(data, th):
    """
    derivative of threshold function!
    """
    I = abs(data)>th

    return I



def AMP(y, A, At=None, lamb=None, iters=None, xi=None, zi=None, xo=None):
    """
    AMP for sparse recovery

    Input:
        y       : the measurements
        A       : the measurement matrix (or a function handle that projects onto A)
        At      : transpose of A (or a function handle that projects onto At)
        lamb       : lambda
        iters   : the number of iterations
        xi      : inital estimate of x
        zi      : inital residue
        xo      : true x (for debugging/monitoring purposes)
    Output:
        x_hat   : the recovered signal.
        MSE    : the MSE trajectory.
    """

    """
    if (nargin >= 5) and (logical_not(isempty(At_func))):
        A=lambda x=None: M_func[x]
        At=lambda z=None: At_func[z]
    else:
        A=lambda x=None: dot(M_func,x)
        At=lambda z=None: dot(M_func.T,z)
    """
    if At is None:
        At = A.T


    """
    M=len(y)
    load('OptimumLambdaSigned.mat')

    delta_check=copy(delta_vec)
    delta = M / N
    lambda_=interp1(delta_check,lambda_opt,delta)

    ## plot optimal delta - lambda tradeoff
    # plot(delta_vec, lambda_opt); xlabel('\delta'); ylabel('\lambda'); box off
    """

    #print(A.shape)
    M,N = A.shape

    ## initial estimate and residue
    if xi is not None:
        xt=xi
        zt=zi
    else:
        xt=np.zeros((N,1))
        zt=y

    if xo is not None:
        MSE = []

    sigma_hat=sqrt(sum(abs(zt)**2)/M)

    sigs = []
    for iter in range(iters):

        # update zt and sigma
        deriv = dEta(dot(At,zt) + xt, lamb*sigma_hat)
        zt = y - dot(A,xt) + zt*sum(deriv)/M

        # update sigma
        sigma_hat=sqrt(sum(abs(zt)**2)/M)
        sigs.append(sigma_hat)  # save sigma for debugging

        # update xt
        xt = eta(dot(At,zt) + xt, lamb*sigma_hat)



        """ old
        pseudo_data=dot(At,zt) + xt  # using z_t-1 and x_t-1

        sigma_hat=sqrt(sum(abs(zt)**2)/M)
        sigs.append(sigma_hat)  # save sigma for debugging

        xt, dEta = eta(pseudo_data, lamb*sigma_hat)
        zt = y - dot(A,xt) + zt*sum(dEta)/M
        """

        if xo is not None:
            MSE.append(sum((xt-xo)**2)/N)

    return xt, zt, sigs, MSE

#if __name__ == '__main__':
#    pass
