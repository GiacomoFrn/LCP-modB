## IMPORT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib.colors import LogNorm
import time

## ADAM ALGORITHM

def adam(grad, init, n_epochs=5000, eta=10**-4, gamma=0.9, beta=0.99,epsilon=10**-8, noise_strength=0, TIME_FLAG=False):
    """
    ADAM algorithm

    Parameters:
    grad           -> callable function that returns the gradient of the function
    init           -> starting values for the search for minimum 
    n_epochs       -> maximum number of iterations
    eta            -> learning rate
    gamma          -> beta1
    beta           -> beta2
    epsilon        -> 
    noise_strength ->
    TIME_FLAG      ->

    Return:
    param_traj     -> trajectory 
    """
    
    if TIME_FLAG: 
        start=time.process_time()
        times=[0]

    params=np.array(init)
    param_traj=np.zeros([n_epochs+1,2])
    param_traj[0,]=init
    v=0;
    grad_sq=0;

    for j in range(1,n_epochs+1):
        noise=noise_strength*np.random.randn(params.size)
        g=np.array(grad(params))+noise
        v=gamma*v+(1-gamma)*g
        grad_sq=beta*grad_sq+(1-beta)*g*g
        v_hat=v/(1-gamma**j)
        grad_sq_hat=grad_sq/(1-beta**j)
        params=params-eta*np.divide(v_hat,np.sqrt(grad_sq_hat)+epsilon)
        param_traj[j,]=params
        if TIME_FLAG: times.append(time.process_time()-start)

    if TIME_FLAG: return param_traj, np.array(times)
    else: return param_traj


## ADAMAX ALGORITHM

def adamax(grad, init, n_epochs=5000, alpha=2e-3, beta1=0.9, beta2=0.999, noise_strength=0, TIME_FLAG=False):
    """
    ADAMAX algorithm

    Parameters:
    grad           -> callable function that returns the gradient of the function
    init           -> starting values for the search for minimum 
    alpha.         -> 
    beta1          -> beta1
    beta2          -> beta2
    noise_strength ->
    TIME_FLAG      ->

    Return:
    param_traj     -> trajectory 
    """

    if TIME_FLAG: 
        start=time.process_time()
        times=[0]

    params=np.array(init)
    param_traj=np.zeros([n_epochs+1,2])
    param_traj[0,]=init
    mt=np.zeros(params.shape)
    ut=np.zeros(params.shape)

    for j in range(1,n_epochs+1):
        noise=noise_strength*np.random.randn(params.size)
        gt=np.array(grad(params))+noise
        mt=beta1*mt+(1-beta1)*gt
        ut=np.amax(np.vstack((beta2*ut, np.absolute(gt))), axis=0)
        params=params-(alpha/(1-beta1**j))*np.divide(mt,ut)
        param_traj[j,]=params
        if TIME_FLAG: times.append(time.process_time()-start)

    if TIME_FLAG: return param_traj, np.array(times)
    else: return param_traj


## VANILLA GRADIENT DESCENT

def gd(grad, init, n_epochs=1000, eta=10**-4, noise_strength=0, TIME_FLAG=False):
    #This is a simple optimizer
    if TIME_FLAG: 
        start=time.process_time()
        times=[0]

    params=np.array(init)
    param_traj=np.zeros([n_epochs+1,2])
    param_traj[0,]=init
    v=0;
    for j in range(n_epochs):
        noise=noise_strength*np.random.randn(params.size)
        v=eta*(np.array(grad(params))+noise)
        params=params-v
        param_traj[j+1,]=params
        if TIME_FLAG: times.append(time.process_time()-start)

    if TIME_FLAG: return param_traj, np.array(times)
    else: return param_traj


## GRADIENT DESCENT WITH MOMENTUM

def gd_with_mom(grad, init, n_epochs=5000, eta=10**-4, gamma=0.9, noise_strength=0, TIME_FLAG=False):
    if TIME_FLAG: 
        start=time.process_time()
        times=[0]
        
    params=np.array(init)
    param_traj=np.zeros([n_epochs+1,2])
    param_traj[0,]=init
    v=0
    for j in range(n_epochs):
        noise=noise_strength*np.random.randn(params.size)
        v=gamma*v+eta*(np.array(grad(params))+noise)
        params=params-v
        param_traj[j+1,]=params
        if TIME_FLAG: times.append(time.process_time()-start)

    if TIME_FLAG: return param_traj, np.array(times)
    else: return param_traj


## NESTEROV

def NAG(grad, init, n_epochs=5000, eta=10**-4, gamma=0.9, noise_strength=0, TIME_FLAG=False):
    
    if TIME_FLAG: 
        start=time.process_time()
        times=[0]
    
    params=np.array(init)
    param_traj=np.zeros([n_epochs+1,2])
    param_traj[0,]=init
    v=0
    for j in range(n_epochs):
        noise=noise_strength*np.random.randn(params.size)
        params_nesterov=params-gamma*v
        v=gamma*v+eta*(np.array(grad(params_nesterov))+noise)
        params=params-v
        param_traj[j+1,]=params
        if TIME_FLAG: times.append(time.process_time()-start)

    if TIME_FLAG: return param_traj, np.array(times)
    else: return param_traj


# RMSPROP

def rms_prop(grad, init, n_epochs=5000, eta=10**-3, beta=0.9, epsilon=10**-8, noise_strength=0, TIME_FLAG=False):
    
    if TIME_FLAG: 
        start=time.process_time()
        times=[0]
    
    params=np.array(init)
    param_traj=np.zeros([n_epochs+1,2])
    param_traj[0,]=init#Import relevant packages
    grad_sq=0;
    for j in range(n_epochs):
        noise=noise_strength*np.random.randn(params.size)
        g=np.array(grad(params))+noise
        grad_sq=beta*grad_sq+(1-beta)*g*g
        v=eta*np.divide(g,np.sqrt(grad_sq+epsilon))
        params= params-v
        param_traj[j+1,]=params
        if TIME_FLAG: times.append(time.process_time()-start)

    if TIME_FLAG: return param_traj, np.array(times)
    else: return param_traj