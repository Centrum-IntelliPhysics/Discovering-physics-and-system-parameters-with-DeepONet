import numpy as np

def L2_ERROR_A(true,pred):
    
    return  np.mean((pred - true) ** 2 / (true ** 2 + 1e-4))     

    
def L2_ERROR_B(true,pred):
    
    return np.linalg.norm(true-pred,2)/np.linalg.norm(true,2)


def R2_SCORE(true_val,pred_val):
    
    mean_true = np.mean(true_val)
    
    return 1.0 - np.mean(np.square(true_val-pred_val))/np.mean(np.square(true_val-mean_true))

def ERROR(true,pred):
    
    return true - pred