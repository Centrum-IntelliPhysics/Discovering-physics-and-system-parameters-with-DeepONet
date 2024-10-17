import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import os
import time as timeR
import scipy.optimize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import warnings
warnings.filterwarnings('ignore')
from models import PI_DeepONet_Darcy_without_Data_sampling_varied
from scipy.interpolate import griddata
from utils import interpolate_k_f_TF, interpolate_k_f, SCALE, RESCALE
from metric import R2_SCORE, ERROR, L2_ERROR_A, L2_ERROR_B


RANDOM_SEED = 2308
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf_datatype = 'float64'
tf.keras.backend.set_floatx(tf_datatype)
np_datatype = np.float64



train_samples = 5000 #5000 #5000
test_samples = 1000 #1000 #1000
val_samples = 10
run_epochs = 5000
col_points = 5000
bc_points = 1024 
input_channel = (31,31,2)
num_filters = [40,60] 
filter_size = [3,3] 
strides_conv = 2
padding = 'same'
strides_pool = 2
pool_size = 2
FC_layers = [64,64,64]
layers = [2,128,128,64]
save_step = 1
val_step = save_step
validate = True
lr_rate = 1e-4
alpha_reg = 0.0
foldername = 'PI_DeepONet_Nolabel'
filename = 'prediction_train_samples_5000'
print('Foldername: ',foldername)
print('Filename: ', filename)

predict_for_train_samples = [5,10]
predict_for_test_samples = [20,30]


def l_shaped_mesh(print_mesh=False):
    if os.path.exists(os.path.join('../../../Data with 2 branches/Data/L_shaped_50.mat')):
        #print('succesfully loaded')
        mat_mesh = scipy.io.loadmat(os.path.join('../../../Data with 2 branches/Data/L_shaped_50.mat'))
        points = mat_mesh['nodes'] + 0.5
        elements = mat_mesh['elms'] - 1
        # points = 0.1 + np.array([[0, 0], [0, 1], [1, 0], [1, 1],]) / 4
        # elements = np.array([[0, 1, 2], [1, 3, 2]])
    else:
        print('not able to locate')

    small_l_bc = np.logical_and(points[:, 0] >= 0.5, points[:, 1] >= 0.5)
    large_l_bc = np.logical_or(points[:, 0] <= 0, points[:, 1] <= 0)
    l_bc = np.logical_or(small_l_bc, large_l_bc)
    influx_bc = points[:, 1] >= 1
    outflux_bc = points[:, 0] >= 1
    l_bc = np.where(influx_bc, False, l_bc)
    l_bc = np.where(outflux_bc, False, l_bc)
    non_bc = np.logical_not(np.logical_or(np.logical_or(l_bc, influx_bc), outflux_bc))

    if print_mesh:
        plt.plot(points[non_bc, 0], points[non_bc, 1], 'o', markersize=1.5)
        plt.plot(points[l_bc, 0], points[l_bc, 1], '*g')
        plt.plot(points[influx_bc, 0], points[influx_bc, 1], '*r')
        plt.plot(points[outflux_bc, 0], points[outflux_bc, 1], '*b')
        plt.triplot(points[:, 0], points[:, 1], elements)
        plt.axis('equal')
        plt.savefig('mesh.png')

    return points, elements, non_bc, l_bc, influx_bc, outflux_bc


def load_data_with_real_k_f():
    # Source model
    with open('../../../Data with 2 branches/Data/processed_data_l.pkl', 'rb') as f:
        file = pickle.load(f)

    s = 31
    r = 450

    k_train = file['k_train'] # (51000, 31, 31)
    f_train = file['f_train'] # (51000, 31, 31)
    u_train = file['u_train'] # (51000, 450)    

    k_test = file['k_test'] #(9000, 31, 31)
    f_test = file['f_test'] #(9000, 31, 31)
    u_test = file['u_test'] #(9000, 450)

    coord_file = scipy.io.loadmat('../../../Data with 2 branches/Data/L_shaped_50')

    xx = coord_file['nodes'][:,0:1] + 0.5 # (450, 1)
    yy = coord_file['nodes'][:,1:2] + 0.5 # (450, 1)
    X = np.hstack((xx, yy)) # (450, 2)

    k_train_real = k_train.copy()
    K_train = np.reshape(k_train, (-1, s, s, 1))  #(51000, 31, 31, 1)
    K_train = (K_train - 1.0)/0.3  # (51000, 31, 31, 1)
    
    k_test_real = k_test.copy()
    K_test = np.reshape(k_test, (-1, s, s, 1))  # (9000, 31, 31, 1)
    K_test = (K_test - 1.0)/0.3  # (9000, 31, 31, 1)
    
    f_train_real = f_train.copy()
    f_norm_train = np.sqrt(np.mean(f_train**2, axis=(1, 2)))[:, None, None, None] #  (51000, 1, 1, 1)
    F_train = np.reshape(f_train, (-1, s, s, 1))/f_norm_train # (51000, 31, 31, 1)
    
    f_test_real = f_test.copy()
    f_norm_test = np.sqrt(np.mean(f_test**2, axis=(1, 2)))[:, None, None, None] #  (9000, 1, 1, 1)
    F_test = np.reshape(f_test, (-1, s, s, 1))/f_norm_test # (9000, 31, 31, 1)

    branch_inp_train = np.concatenate((K_train,F_train), axis = -1) # (51000, 31, 31, 2)
    branch_inp_test = np.concatenate((K_test,F_test), axis = -1)  # (9000, 31, 31, 2)
    U_train = np.reshape(u_train, (-1, r, 1)) # (51000, 450, 1)
    U_test = np.reshape(u_test, (-1, r, 1))  # (9000, 450, 1)

    return branch_inp_train, U_train, branch_inp_test, U_test, X, f_norm_train, f_norm_test, k_train_real, k_test_real, f_train_real, f_test_real





def generate_train_test_samples_with_real_k_f(train_samples,test_samples):
    
    branch_inp_train_load, U_train_load, branch_inp_test_load, U_test_load, X_load, f_norm_train_load, f_norm_test_load, k_train_real_load, k_test_real_load, f_train_real_load, f_test_real_load = load_data_with_real_k_f()
    
    branch_inp_train = branch_inp_train_load[0:train_samples,:,:,:]
    U_train = U_train_load[0:train_samples,:,:]
    branch_inp_test = branch_inp_test_load[0:test_samples,:,:,:]
    U_test = U_test_load[0:test_samples,:,:]
    X = X_load
    f_norm_train = f_norm_train_load[0:train_samples,:,:,:]
    f_norm_test = f_norm_test_load[0:test_samples,:,:,:]
    
    
    k_train_real = k_train_real_load[0:train_samples,:,:]
    f_train_real = f_train_real_load[0:train_samples,:,:]
    k_test_real = k_test_real_load[0:test_samples,:,:]
    f_test_real = f_test_real_load[0:test_samples,:,:]
    
    return branch_inp_train, U_train, branch_inp_test, U_test, X, f_norm_train, f_norm_test, k_train_real, k_test_real, f_train_real, f_test_real





test_val_samples = test_samples + val_samples

branch_inp_train, U_train, branch_inp_test_val, U_test_val, X, f_norm_train, f_norm_test_val, k_train_real, k_test_val_real, f_train_real, f_test_val_real  = generate_train_test_samples_with_real_k_f(train_samples,test_val_samples)

branch_inp_test = branch_inp_test_val[0:test_samples,:,:,:]
branch_inp_val = branch_inp_test_val[test_samples:,:,:,:]
U_test = U_test_val[0:test_samples,:,:]
U_val = U_test_val[test_samples:,:,:]
f_norm_test = f_norm_test_val[0:test_samples,:,:,:]
f_norm_val = f_norm_test_val[test_samples:,:,:,:]


k_test_real =  k_test_val_real[0:test_samples,:,:]
k_val_real =  k_test_val_real[test_samples:,:,:]
f_test_real =  f_test_val_real[0:test_samples,:,:]
f_val_real =  f_test_val_real[test_samples:,:,:]



# print('branch_inp_train: ',branch_inp_train.shape)
# print('branch_inp_test: ',branch_inp_test.shape)
# print('branch_inp_val: ',branch_inp_val.shape)
# print('U_train: ',U_train.shape)
# print('U_test: ',U_test.shape)
# print('U_val: ',U_val.shape)
# print('f_norm_train: ',f_norm_train.shape)
# print('f_norm_test: ',f_norm_test.shape)
# print('f_norm_val: ',f_norm_val.shape)
# print('X: ',X.shape)
# print('k_train_real: ',k_train_real.shape)
# print('k_test_real: ',k_test_real.shape)
# print('k_val_real: ',k_val_real.shape)
# print('f_train_real: ',f_train_real.shape)
# print('f_test_real: ',f_test_real.shape)
# print('f_val_real: ',f_val_real.shape)

points, simplices, _, _, _, _ = l_shaped_mesh()

Nx_interp, Ny_interp = 31, 31
x_grid = np.linspace(0.0,1.0,31)
y_grid = np.linspace(0.0,1.0,31)
xx_grid, yy_grid = np.meshgrid(x_grid,y_grid)
X_grid = np.concatenate((xx_grid.flatten()[:,None],yy_grid.flatten()[:,None]),1)



input_trunk_train = np.array([],np_datatype).reshape(-1,2)
target_train = np.array([],np_datatype).reshape(-1,1)
#k_train =  np.array([],np_datatype).reshape(-1,1)
#f_train =  np.array([],np_datatype).reshape(-1,1)

for i in range(train_samples):

    #X_grid, u_interp =  interpolate_u(X,U_train[i].flatten(),Nx_interp, Ny_interp,method = interpolate_scheme)
    
    input_trunk_train_arr = X #X_grid
    target_train_arr = U_train[i] #u_interp[:,None]
    #k_train_arr  = k_train_real[i].flatten()[:,None]
    #f_train_arr  = f_train_real[i].flatten()[:,None]
    
    
    input_trunk_train = np.concatenate((input_trunk_train,input_trunk_train_arr),axis = 0)
    target_train = np.concatenate((target_train,target_train_arr),axis=0)
    #k_train = np.concatenate((k_train,k_train_arr),axis=0)
    #f_train = np.concatenate((f_train,f_train_arr),axis=0)

input_branch_train = branch_inp_train*1.0

k_train = k_train_real*1.0
f_train = f_train_real*1.0

for i in range(len(k_train)):
    k_train[i] = k_train[i].T
    f_train[i] = f_train[i].T
    
    

input_trunk_val = np.array([],np_datatype).reshape(-1,2)
target_val = np.array([],np_datatype).reshape(-1,1)
#k_val =  np.array([],np_datatype).reshape(-1,1)
#f_val =  np.array([],np_datatype).reshape(-1,1)

for i in range(val_samples):

    #X_grid, u_interp =  interpolate_u(X,U_val[i].flatten(),Nx_interp, Ny_interp,method = interpolate_scheme)
    
    input_trunk_val_arr = X #X_grid
    target_val_arr = U_val[i] #u_interp[:,None]
    #k_val_arr  = k_val_real[i].flatten()[:,None]
    #f_val_arr  = f_val_real[i].flatten()[:,None]
    
    
    input_trunk_val = np.concatenate((input_trunk_val,input_trunk_val_arr),axis = 0)
    target_val = np.concatenate((target_val,target_val_arr),axis=0)
    #k_val = np.concatenate((k_val,k_val_arr),axis=0)
    #f_val = np.concatenate((f_val,f_val_arr),axis=0)
    
input_branch_val = branch_inp_val*1.0

k_val = k_val_real*1.0
f_val = f_val_real*1.0


for i in range(len(k_val)):
    k_val[i] = k_val[i].T
    f_val[i] = f_val[i].T
    
    
#std_target, mean_target = np.std(target_train), np.mean(target_train)

std_target = 0.008369216500427129
mean_target = -2.3589802947842785e-05

target_train_scale = SCALE(target_train,mean_target,std_target)
target_val_scale = SCALE(target_val,mean_target,std_target)


batch_size_train = len(X)
batch_size_val = len(X)


model =  PI_DeepONet_Darcy_without_Data_sampling_varied(None,input_branch_train,None,k_train,f_train,None,validate,input_trunk_val,input_branch_val,target_val_scale,batch_size_val,col_points,bc_points,mean_target,std_target,layers,input_channel,num_filters,filter_size,strides_conv,padding,strides_pool,pool_size,FC_layers,lr_rate,save_step)   
        
model.ADAM_optimizer.lr.assign(1e-4)
model.train_model(run_epochs)

itr_loss = np.arange(1,len(model.loss_net_train_list)+1)*save_step
itr_loss[0] = 1
plt.figure()
plt.loglog(itr_loss,model.loss_net_train_list,c='r')
plt.xlabel('Epoch',fontsize=15)
plt.ylabel('Loss',fontsize=15)
plt.grid('True')
plt.tight_layout()
fname = foldername + '/loss_total_' + filename + '.png'
plt.savefig(fname)
#plt.show()


plt.figure()
plt.loglog(itr_loss,model.loss_net_train_list,label='Loss: train',c='r')
if validate:
    plt.loglog(itr_loss,model.loss_net_val_list,label='Loss: val',c='b')
plt.xlabel('Epoch',fontsize=15)
plt.ylabel('Loss',fontsize=15)
plt.grid('True')
plt.legend(fontsize=12)
plt.tight_layout()
fname = foldername +  '/loss_validation_' + filename + '.png'
plt.savefig(fname)
#plt.show()

weights_dict = {}
weights_final = model.train_variable_list()
 
for i in range(len(weights_final)):
    weights_dict['w' + str(i)] = weights_final[i].numpy()
    

wfile = foldername + '/weights_' + filename  + '.mat'
scipy.io.savemat(wfile,weights_dict)    




l2_error_a_train = []
r2_score_train = []

u_true_train = np.array([]).reshape(0,1) 
u_pred_train = np.array([]).reshape(0,1) 

train_samples_caln  = train_samples #min(train_samples,1000)

for i in range(train_samples_caln):
    u_pred = model.predict(X,branch_inp_train[i:i+1,:])
    u_pred = RESCALE(u_pred,mean_target,std_target)
    u_true = U_train[i,:]
    l2_error_a = L2_ERROR_A(u_true.flatten(),u_pred.flatten())
    r2_score = R2_SCORE(u_true.flatten(),u_pred.flatten())
    l2_error_a_train.append(l2_error_a)
    r2_score_train.append(r2_score)
    u_true_train = np.concatenate((u_true_train,u_true),axis=0)
    u_pred_train = np.concatenate((u_pred_train,u_pred),axis=0)


l2_error_a_test = []
r2_score_test = []

u_true_test = np.array([]).reshape(0,1) 
u_pred_test = np.array([]).reshape(0,1) 

for i in range(test_samples):
    u_pred = model.predict(X,branch_inp_test[i:i+1,:])
    u_pred = RESCALE(u_pred,mean_target,std_target)
    u_true = U_test[i,:]
    l2_error_a = L2_ERROR_A(u_true.flatten(),u_pred.flatten())
    r2_score = R2_SCORE(u_true.flatten(),u_pred.flatten())
    l2_error_a_test.append(l2_error_a)
    r2_score_test.append(r2_score)
    u_true_test = np.concatenate((u_true_test,u_true),axis=0)
    u_pred_test = np.concatenate((u_pred_test,u_pred),axis=0)
    
    
    
l2_error_a_train_all = L2_ERROR_A(u_true_train.flatten(),u_pred_train.flatten())
l2_error_a_test_all = L2_ERROR_A(u_true_test.flatten(),u_pred_test.flatten())

print('Train L2 error A (all samples): ',l2_error_a_train_all)
print('Test L2 error A (all samples): ',l2_error_a_test_all)


plt.figure(figsize=(12,4))
plt.subplot(121)
plt.hist(l2_error_a_train,color='b',label='train',bins=50)
plt.ylabel('# Samples')
plt.xlabel(r'$L_2$' + ' Error A')
plt.legend()
plt.subplot(122)
plt.hist(l2_error_a_test,color='g',label='test',bins=50)
plt.ylabel('# Samples')
plt.xlabel(r'$L_2$' + ' Error A')
plt.legend()
plt.tight_layout()
fname = foldername + '/dist_error_' + filename + '.png'
plt.savefig(fname)
#plt.show()

print('Train L2 error A mean: ',np.mean(l2_error_a_train))
print('Train L2 error A std. dev. : ',np.std(l2_error_a_train))
print('Test L2 error A mean: ',np.mean(l2_error_a_test))
print('Test L2 error A std. dev. : ',np.std(l2_error_a_test))


    
def lhs_rhs_eqn_pred(input_trunk,input_branch,k,f):

    x, y = input_trunk[:,0:1], input_trunk[:,1:2]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)
        u_scaled  = model.model_output(tf.stack([x[:,0],y[:,0]], axis=1),input_branch)
        u = RESCALE(u_scaled,model.mean_target,model.std_target)
        
        u_x = tape.gradient(u,x)
        u_y = tape.gradient(u,y)

        k_times_u_x = k*u_x
        k_times_u_y = k*u_y
        
    k_times_u_x_x = tape.gradient(k_times_u_x,x)
    k_times_u_y_y = tape.gradient(k_times_u_y,y)
    

    LHS = k_times_u_x_x +  k_times_u_y_y 
    RHS = - f
    
    del tape

    return LHS, RHS


def interpolate_LHS_RHS(LHS,RHS,X,method='linear'):
    
    lhs_interp = griddata(X_grid,LHS,X,method=method)
    rhs_interp = griddata(X_grid,RHS,X,method=method)
    
    return lhs_interp, rhs_interp



def plot_soln_for_train_sample(sample_no,SAVE_PLOT=True):

    i = sample_no
    u_pred = model.predict(X,branch_inp_train[i:i+1,:])
    u_pred = u_pred.flatten()[:,None]
    u_pred = RESCALE(u_pred,mean_target,std_target).flatten()
    u_true = U_train[i,:].flatten()

    fig, Ax = plt.subplots(nrows=1, ncols=3,figsize=(14,4))
    u_plot = u_true
    ax = Ax[0]
    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, u_plot,
                       levels=np.linspace(np.min(u_plot), np.max(u_plot), 100),cmap='jet')

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ticks = np.linspace(np.min(u_plot),np.max(u_plot),8)
    cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.tick_right()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('Train sample '+str(sample_no) + ': True')
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal', 'box')

    u_plot = u_pred
    ax = Ax[1]
    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, u_plot,
                       levels=np.linspace(np.min(u_plot), np.max(u_plot), 100),cmap='jet')

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ticks = np.linspace(np.min(u_plot),np.max(u_plot),8)
    cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.tick_right()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('Pred')
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal', 'box')


    u_plot = np.abs(u_true - u_pred)
    ax = Ax[2]
    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, u_plot,
                       levels=np.linspace(np.min(u_plot), np.max(u_plot), 100),cmap='jet')

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ticks = np.linspace(np.min(u_plot),np.max(u_plot),8)
    cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.tick_right()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('Abs. Error')
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    if SAVE_PLOT:
        fname =foldername + '/sample_train_' + str(sample_no) + '_' + filename + '.png'
        plt.savefig(fname)
    #plt.show()
    l2_error_a = L2_ERROR_A(u_true,u_pred)
    r2_score = R2_SCORE(u_true,u_pred)

    print('Training sample ' + str(sample_no) + ', L2 error A: ',l2_error_a,', R2 score: ',r2_score)

def plot_eqn_for_train_sample(sample_no,SAVE_PLOT=True):

    i = sample_no
    LHS_pred, RHS_true = lhs_rhs_eqn_pred(tf.convert_to_tensor(X_grid),tf.convert_to_tensor(branch_inp_train[i:i+1,:]),k_train_real[i].T.flatten()[:,None],f_train_real[i].T.flatten()[:,None])

    LHS_pred = LHS_pred.numpy().reshape(Ny_interp,Nx_interp)
    RHS_pred = RHS_true.reshape(Ny_interp,Nx_interp)

    LHS_pred, RHS_pred = interpolate_LHS_RHS(LHS_pred.flatten(),RHS_pred.flatten(),X)


    fig, Ax = plt.subplots(nrows=1, ncols=3,figsize=(14,4))
    u_plot = RHS_pred
    ax = Ax[0]
    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, u_plot,
                       levels=np.linspace(np.min(u_plot), np.max(u_plot), 100),cmap='jet')

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ticks = np.linspace(np.min(u_plot),np.max(u_plot),8)
    cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.tick_right()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('Train sample '+str(sample_no) + ': RHS True')
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal', 'box')

    u_plot = LHS_pred
    ax = Ax[1]
    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, u_plot,
                       levels=np.linspace(np.min(u_plot), np.max(u_plot), 100),cmap='jet')

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ticks = np.linspace(np.min(u_plot),np.max(u_plot),8)
    cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.tick_right()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('LHS: Pred')
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal', 'box')


    u_plot = np.abs(LHS_pred - RHS_pred)
    ax = Ax[2]
    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, u_plot,
                       levels=np.linspace(np.min(u_plot), np.max(u_plot), 100),cmap='jet')

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ticks = np.linspace(np.min(u_plot),np.max(u_plot),8)
    cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.tick_right()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('Abs. Error')
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    if SAVE_PLOT:
        fname = foldername + '/sample_train_' + str(sample_no) + '_eqn_' +  filename + '.png'
        plt.savefig(fname)
    #plt.show()




def plot_soln_for_test_sample(sample_no,SAVE_PLOT=True):


    i = sample_no
    u_pred = model.predict(X,branch_inp_test[i:i+1,:])
    u_pred = u_pred.flatten()[:,None]
    u_pred = RESCALE(u_pred,mean_target,std_target).flatten()
    u_true = U_test[i,:].flatten()


    fig, Ax = plt.subplots(nrows=1, ncols=3,figsize=(14,4))
    u_plot = u_true
    ax = Ax[0]
    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, u_plot,
                       levels=np.linspace(np.min(u_plot), np.max(u_plot), 100),cmap='jet')

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ticks = np.linspace(np.min(u_plot),np.max(u_plot),8)
    cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.tick_right()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('Test sample '+str(sample_no) + ': True')
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal', 'box')

    u_plot = u_pred
    ax = Ax[1]
    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, u_plot,
                       levels=np.linspace(np.min(u_plot), np.max(u_plot), 100),cmap='jet')

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ticks = np.linspace(np.min(u_plot),np.max(u_plot),8)
    cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.tick_right()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('Pred')
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal', 'box')


    u_plot = np.abs(u_true - u_pred)
    ax = Ax[2]
    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, u_plot,
                       levels=np.linspace(np.min(u_plot), np.max(u_plot), 100),cmap='jet')

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ticks = np.linspace(np.min(u_plot),np.max(u_plot),8)
    cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.tick_right()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('Abs. Error')
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    if SAVE_PLOT:
        fname = foldername + '/sample_test_' + str(sample_no) + '_' + filename + '.png'
        plt.savefig(fname)
    #plt.show()
    l2_error_a = L2_ERROR_A(u_true,u_pred)
    r2_score = R2_SCORE(u_true,u_pred)

    print('Testing sample ' + str(sample_no) + ', L2 error A: ',l2_error_a,', R2 score: ',r2_score)

def plot_eqn_for_test_sample(sample_no,SAVE_PLOT=True):

    i = sample_no
    LHS_pred, RHS_true = lhs_rhs_eqn_pred(tf.convert_to_tensor(X_grid),tf.convert_to_tensor(branch_inp_test[i:i+1,:]),k_test_real[i].T.flatten()[:,None],f_test_real[i].T.flatten()[:,None])

    LHS_pred = LHS_pred.numpy().reshape(Ny_interp,Nx_interp)
    RHS_pred = RHS_true.reshape(Ny_interp,Nx_interp)

    LHS_pred, RHS_pred = interpolate_LHS_RHS(LHS_pred.flatten(),RHS_pred.flatten(),X)

    fig, Ax = plt.subplots(nrows=1, ncols=3,figsize=(14,4))
    u_plot = RHS_pred
    ax = Ax[0]
    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, u_plot,
                       levels=np.linspace(np.min(u_plot), np.max(u_plot), 100),cmap='jet')

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ticks = np.linspace(np.min(u_plot),np.max(u_plot),8)
    cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.tick_right()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('Test sample '+str(sample_no) + ': RHS True')
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal', 'box')

    u_plot = LHS_pred
    ax = Ax[1]
    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, u_plot,
                       levels=np.linspace(np.min(u_plot), np.max(u_plot), 100),cmap='jet')

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ticks = np.linspace(np.min(u_plot),np.max(u_plot),8)
    cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.tick_right()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('LHS: Pred')
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal', 'box')


    u_plot = np.abs(LHS_pred - RHS_pred)
    ax = Ax[2]
    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, u_plot,
                       levels=np.linspace(np.min(u_plot), np.max(u_plot), 100),cmap='jet')

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    ticks = np.linspace(np.min(u_plot),np.max(u_plot),8)
    cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.tick_right()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('Abs. Error')
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    if SAVE_PLOT:
        fname = foldername + '/sample_test_' + str(sample_no) + '_eqn_' +  filename + '.png'
        plt.savefig(fname)

    #plt.show()



    

info_dict = {'std_target':std_target, 'mean_target':mean_target,
            'itr_loss':itr_loss,'loss_net_train_list':model.loss_net_train_list,
             'loss_net_val_list':model.loss_net_val_list,'l2_error_a_train':l2_error_a_train,
             'r2_score_train':r2_score_train,'l2_error_a_test':l2_error_a_test,'r2_score_test':r2_score_test,
             'l2_error_a_train_all':l2_error_a_train_all,'l2_error_a_test_all':l2_error_a_test_all,
             'train_samples':train_samples,'test_samples':test_samples,'val_samples':val_samples,'run_epochs':run_epochs,
             'num_filters':num_filters,'filter_size':filter_size,'strides_conv':strides_conv,
             'padding':padding,'strides_pool':strides_pool,'pool_size':pool_size,'FC_layers':FC_layers,
             'layers':layers,'save_step':save_step,'val_step':val_step,'validate':validate,
             'batch_size_train':batch_size_train,'batch_size_val':batch_size_val,'filename':filename,
             'foldername':foldername,'alpha_reg':alpha_reg, 'col_points':col_points,'bc_points': bc_points
            }

info_file = foldername + '/info_' + filename  + '.mat'
scipy.io.savemat(info_file,info_dict) 


for sample_no in predict_for_train_samples:    
    sample_no_train  = sample_no
    plot_soln_for_train_sample(sample_no_train)
    plot_eqn_for_train_sample(sample_no_train)


for sample_no in predict_for_test_samples:    
    sample_no_test  = sample_no    
    plot_soln_for_test_sample(sample_no_test)
    plot_eqn_for_test_sample(sample_no_test)

