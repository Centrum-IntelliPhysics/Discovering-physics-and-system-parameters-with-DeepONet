import numpy as np
from pyDOE import lhs
import tensorflow as tf
from scipy.interpolate import griddata

#lb_domain = X.min(0)
#ub_domain = X.max(0)

lb_domain = np.array([0., 0.])
ub_domain = np.array([1., 1.])

lb_domain1 = lb_domain*1.0
ub_domain1 = ub_domain*1.0
ub_domain1[1] = 0.5

lb_domain2 = lb_domain*1.0
ub_domain2 = ub_domain*1.0
lb_domain2[1] = 0.5
ub_domain2[0] = 0.5


def SCALE(x,mean,std):

    x_scale = (x - mean)/std
    
    return x_scale

def RESCALE(x,mean,std):
    
    x_rescale = x*std + mean
    
    return x_rescale


def sample_collocation_points(N):
    
    N_domain1 = int(2/3*N)
    N_domain2 = N - N_domain1 

    X1 = lb_domain1 + (ub_domain1 -lb_domain1)*lhs(2,N_domain1)
    X2 = lb_domain2 + (ub_domain2 -lb_domain2)*lhs(2,N_domain2)
    X = np.concatenate((X1,X2),0)
    
    X_tf = tf.convert_to_tensor(X)
    
    return X_tf

def sample_boundary_points(N,mean_target,std_target):
    
    N_boundary1 = N//4 
    N_boundary2 = N//8
    N_boundary3 = N//8
    N_boundary4 = N//8
    N_boundary5 = N//8
    N_boundary6 = N//4

    X1 = np.random.rand(N_boundary1,2)
    X1[:,0] = 0.0
    X2 = 0.5*np.random.rand(N_boundary2,2)
    X2[:,1] = 1.0
    X3 = 0.5*np.random.rand(N_boundary3,2) + 0.5 
    X3[:,0] = 0.5
    X4 = 0.5*np.random.rand(N_boundary4,2) + 0.5 
    X4[:,1] = 0.5
    X5 = 0.5*np.random.rand(N_boundary5,2)  
    X5[:,0] = 1.0
    X6 = np.random.rand(N_boundary6,2)  
    X6[:,1] = 0.0

    X = np.concatenate((X1,X2,X3,X4,X5,X6),0)

    X = tf.convert_to_tensor(X)
    
    u = X[:,0:1]*0.0
    
    u = SCALE(u,mean_target,std_target)
    
    return X , u



Nx_interp, Ny_interp = 31, 31
x_grid = np.linspace(0.0,1.0,31)
y_grid = np.linspace(0.0,1.0,31)
xx_grid, yy_grid = np.meshgrid(x_grid,y_grid)
X_grid = np.concatenate((xx_grid.flatten()[:,None],yy_grid.flatten()[:,None]),1)


def interpolate_k_f(k,f,X,method='linear'):
    
    k_interp = griddata(X_grid,k,X,method=method)
    f_interp = griddata(X_grid,f,X,method=method)
    
    return k_interp, f_interp


def interpolate_k_f_TF(k,f,X,method='linear'):
    
    k_interp = griddata(X_grid,k,X,method=method)
    f_interp = griddata(X_grid,f,X,method=method)
    
    return tf.convert_to_tensor(k_interp[:,None]), tf.convert_to_tensor(f_interp[:,None])




