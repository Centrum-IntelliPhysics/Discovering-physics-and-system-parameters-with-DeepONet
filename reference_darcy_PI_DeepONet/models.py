import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import time as timeR
import warnings
warnings.filterwarnings('ignore')
from utils import sample_collocation_points, sample_boundary_points, interpolate_k_f_TF, SCALE, RESCALE


tf_datatype = 'float64'
tf.keras.backend.set_floatx(tf_datatype)
np_datatype = np.float64


class PI_DeepONet_Darcy_with_Data_sampling_varied:
    
    def __init__(self,input_trunk_train,input_branch_train,target_train,k_train,f_train,batch_size_train,validate,input_trunk_val,input_branch_val,target_val,batch_size_val,col_points,bc_points,mean_target,std_target,layers,input_channel,num_filters,filter_size,strides_conv,padding,strides_pool,pool_size,FC_layers,lr_rate,save_step):   
        
        self.input_trunk_train = input_trunk_train
        self.input_branch_train = input_branch_train
        self.target_train = target_train
        self.k_train = k_train
        self.f_train = f_train
        self.batch_size_train = batch_size_train
        self.input_trunk_val = input_trunk_val
        self.input_branch_val = input_branch_val
        self.target_val  = target_val
        self.batch_size_val = batch_size_val
        self.col_points = col_points 
        self.bc_points = bc_points
        #self.val_step = val_step
        self.lb_trunk = [] #lb_trunk
        self.ub_trunk = [] #ub_trunk
        self.layers = layers
        self.input_channel = input_channel
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.strides_conv = strides_conv
        self.strides_pool = strides_pool
        self.padding = padding
        self.pool_size = pool_size
        self.FC_layers = FC_layers
        self.lr_rate = lr_rate 
        self.save_step = save_step
        self.validate = validate
        self.mean_target = mean_target
        self.std_target = std_target
        self.loss_net_train_list = []
        self.loss_net_val_list = []
        self.save_step = save_step
        self.iters = 0
        self.training_time_in_min = 0.0  
        self.training_time_per_epoch_in_min = 0.0
        self.iters_list = []
        self.training_time_in_min = None
        self.training_time_per_itr_in_sec = None
        
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.input_trunk_train,self.target_train)).batch(self.batch_size_train)
        
        self.steps_train =  len(self.train_dataset)
        
        if self.validate:
            self.val_dataset = tf.data.Dataset.from_tensor_slices((self.input_trunk_val,self.target_val)).batch(self.batch_size_val)
            self.steps_val =  len(self.val_dataset)
        
        self.beta = 0.0

        #self.bias = tf.Variable(tf.zeros([1,1], dtype=tf_datatype), dtype=tf_datatype)
        
        self.model_trunk = self.model_trunk_net(self.layers,self.lb_trunk,self.ub_trunk)
        
        self.model_branch = self.model_branch_net(self.input_channel,self.num_filters,self.filter_size,self.strides_conv,self.padding,self.strides_pool,self.pool_size,self.FC_layers)
        
        self.len_trunk_w = len(self.model_trunk.trainable_variables)//2
        
        self.ADAM_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_rate)
            

    def model_trunk_net(self,layers,lb_trunk,ub_trunk):
        
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(layers[0]))
        #scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0*(x - lb_trunk)/(ub_trunk - lb_trunk) - 1.0)
        #model.add(scaling_layer)
        for i in range(1,len(layers)-1):
            model.add(tf.keras.layers.Dense(layers[i],activation=tf.keras.activations.get('tanh'), kernel_initializer='glorot_normal'))
        model.add(tf.keras.layers.Dense(layers[-1],use_bias=True,activation=tf.keras.activations.get('linear'), kernel_initializer='glorot_normal'))
        
        return model
    

    def model_branch_net(self,input_channel,num_filters,filter_size,strides_conv,padding,strides_pool,pool_size,FC_layers):

        model = tf.keras.models.Sequential()

        for j in range(len(num_filters)):
            
            if j==0:
                model.add(tf.keras.layers.Conv2D(filters = num_filters[j],kernel_size = filter_size[j], strides = strides_conv, padding = padding, activation='relu', input_shape = input_channel))
            else:
                model.add(tf.keras.layers.Conv2D(filters = num_filters[j],kernel_size = filter_size[j], strides = strides_conv, padding = padding, activation='relu'))

            model.add(tf.keras.layers.MaxPooling2D(pool_size = pool_size, strides = strides_pool))

        model.add(tf.keras.layers.Flatten())

        for i in range(len(FC_layers)-1):
            model.add(tf.keras.layers.Dense(FC_layers[i], activation=tf.keras.activations.get('relu'), kernel_initializer='glorot_normal'))
        model.add(tf.keras.layers.Dense(FC_layers[-1], use_bias=True,activation=tf.keras.activations.get('linear'),
                                        kernel_initializer='glorot_normal'))

        return model
    

    def model_output(self,input_trunk,input_branch):
        
        out_trunk  = self.model_trunk(input_trunk)
        out_branch  =  self.model_branch(input_branch)
        G = out_branch*out_trunk
        output = tf.reduce_sum(G,axis=1,keepdims= True)  #+ self.bias
      
        return output
    
    
    def equation_residual(self,input_trunk,input_branch,k,f):
        
        x, y = input_trunk[:,0:1], input_trunk[:,1:2]
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            u_scaled  = self.model_output(tf.stack([x[:,0],y[:,0]], axis=1),input_branch)
            u = RESCALE(u_scaled,self.mean_target,self.std_target)
            
            u_x = tape.gradient(u,x)
            u_y = tape.gradient(u,y)
            
            k_times_u_x = k*u_x
            k_times_u_y = k*u_y
            
        
        k_times_u_x_x = tape.gradient(k_times_u_x,x)
        k_times_u_y_y = tape.gradient(k_times_u_y,y)
        
        residual = k_times_u_x_x +  k_times_u_y_y + f
        
        del tape
        
        return residual
        
    
    def loss_calculate(self,input_branch,Train_data,BC_data,col_data,k,f):
        
        input_trunk_data, target_data = Train_data
        
        input_trunk_bc, target_bc = BC_data
        
        input_trunk_col = col_data
        
        pred_data  = self.model_output(input_trunk_data,input_branch)
        pred_bc  = self.model_output(input_trunk_bc,input_branch)
        
        eqn_residual = self.equation_residual(input_trunk_col,input_branch,k,f)
        
        loss_data = tf.reduce_mean(tf.square(pred_data - target_data))
        loss_bc = tf.reduce_mean(tf.square(pred_bc - target_bc))
        loss_eqn =  tf.reduce_mean(tf.square(eqn_residual)) 

        #sum_w = self.reguralizer_sum_w()
        #loss_reg = self.beta*sum_w
        
        loss_net = loss_data + loss_bc + loss_eqn  #+ loss_reg
                          
        #loss_info = (loss, loss_reg)
        
        return loss_net
    
    def loss_validate_calculate(self,input_trunk,input_branch,target):
        
        pred  = self.model_output(input_trunk,input_branch)
        loss = tf.reduce_mean(tf.square(pred - target))

        return loss
    

    def train_variable_list(self):
        
        train_var = []
        train_var.extend(self.model_branch.trainable_variables)
        train_var.extend(self.model_trunk.trainable_variables)
        #train_var.extend([self.bias])
        
        return train_var
 
    
    def get_grad(self,input_branch,Train_data,BC_data,col_data,k,f):
        
        with tf.GradientTape() as tape:
            tape.watch(self.train_variable_list())           
            loss_net = self.loss_calculate(input_branch,Train_data,BC_data,col_data,k,f)
            
        grad_loss_net = tape.gradient(loss_net, self.train_variable_list())
        del tape

        return loss_net, grad_loss_net
    
    def reguralizer_sum_w(self):

        sum_w_trunk = 0
        for i in range(self.len_trunk_w):
            sum_w_trunk = sum_w_trunk + tf.reduce_sum(tf.square(self.model_trunk.trainable_variables[2*i]))
      
        sum_w = sum_w_trunk
        
        return sum_w
    
    
    def loss_calculate_val(self):
        
        val_dataset_iter = iter(self.val_dataset)
        loss_net_avg  = 0.0
        steps = self.steps_val
        for i in range(steps):
            input_trunk, target = next(val_dataset_iter)
            input_branch =  self.input_branch_val[i:i+1]
            loss_net = self.loss_validate_calculate(input_trunk,input_branch,target)
            loss_net_avg = loss_net_avg  + loss_net
            
        loss_net_avg = loss_net_avg/steps

        return loss_net_avg

        
    def train_with_AdamOptimizer(self,iteration):
                
        @tf.function
        def train_step(input_branch,Train_data,BC_data,col_data,k,f):
            loss_net, grad_loss_net =  self.get_grad(input_branch,Train_data,BC_data,col_data,k,f)
            self.ADAM_optimizer.apply_gradients(zip(grad_loss_net, self.train_variable_list()))

            return loss_net

        
        def train_epoch():
            
            train_dataset_iter = iter(self.train_dataset)
            loss_net_avg  = 0.0
            steps = self.steps_train
            
            for i in range(steps):
                
                input_trunk_data, target_data  = next(train_dataset_iter)
      
                input_branch =  self.input_branch_train[i:i+1]
                
                col_data = sample_collocation_points(self.col_points)  #input_trunk_data # 
                
                k, f = interpolate_k_f_TF(self.k_train[i].flatten(),self.f_train[i].flatten(),col_data.numpy())
        
                input_trunk_bc, target_bc = sample_boundary_points(self.bc_points,self.mean_target,self.std_target)    
                
                Train_data = (input_trunk_data,target_data)
                BC_data = (input_trunk_bc, target_bc)
                
                loss_net = train_step(input_branch,Train_data,BC_data,col_data,k,f)
                          
                loss_net_avg = loss_net_avg + loss_net
            
            loss_net_avg = loss_net_avg/steps
        
            return loss_net_avg
            
        print('Number of steps in one Epoch : ',self.steps_train)
        
        for itr in range(iteration):
                          
            loss_net_train_ = train_epoch()
            
            if self.iters % self.save_step ==0:
                loss_net_train = loss_net_train_.numpy()
                self.loss_net_train_list.append(loss_net_train)
                print('Epoch : ',self.iters ,' Loss train : ',loss_net_train, end=' ')
            
                if self.validate:
                        loss_net_val_ = self.loss_calculate_val()
                        loss_net_val = loss_net_val_.numpy()         
                        self.loss_net_val_list.append(loss_net_val)
                        print(' Loss val : ', loss_net_val)
                      
            self.iters = self.iters + 1 
            
            
    def train_model(self,iteration):
                          
        if self.iters == 0:
            print('-'*100)
            print("Trunk Network")
            self.model_trunk.summary()
            print('-'*100)
            print("Branch Network")
            self.model_branch.summary()
            print('-'*100)
                          
        print('Adam Optimization Starts')
        time_start = timeR.time()
        self.train_with_AdamOptimizer(iteration)
        time_end = timeR.time()
        time_elapsed = time_end - time_start
        self.training_time_in_min = (time_elapsed)/60.0
        self.training_time_per_epoch_in_min = self.training_time_in_min/iteration
        print('Adam Optimization Ends')
        print("Training Time in min for number of Epochs ",iteration," : ",self.training_time_in_min)
                          
        print('-'*100)
     
    def predict(self,input_trunk,input_branch):
        
        pred = self.model_output(input_trunk,input_branch)
        
        return pred.numpy()
    
    
    


class PI_DeepONet_Darcy_without_Data_sampling_varied:
    
    def __init__(self,input_trunk_train,input_branch_train,target_train,k_train,f_train,batch_size_train,validate,input_trunk_val,input_branch_val,target_val,batch_size_val,col_points,bc_points,mean_target,std_target,layers,input_channel,num_filters,filter_size,strides_conv,padding,strides_pool,pool_size,FC_layers,lr_rate,save_step):   
        
        self.input_trunk_train = input_trunk_train
        self.input_branch_train = input_branch_train
        self.target_train = target_train
        self.k_train = k_train
        self.f_train = f_train
        self.batch_size_train = batch_size_train
        self.input_trunk_val = input_trunk_val
        self.input_branch_val = input_branch_val
        self.target_val  = target_val
        self.batch_size_val = batch_size_val
        self.col_points = col_points 
        self.bc_points = bc_points
        #self.val_step = val_step
        self.lb_trunk = [] #lb_trunk
        self.ub_trunk = [] #ub_trunk
        self.layers = layers
        self.input_channel = input_channel
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.strides_conv = strides_conv
        self.strides_pool = strides_pool
        self.padding = padding
        self.pool_size = pool_size
        self.FC_layers = FC_layers
        self.lr_rate = lr_rate 
        self.save_step = save_step
        self.validate = validate
        self.mean_target = mean_target
        self.std_target = std_target
        self.loss_net_train_list = []
        self.loss_net_val_list = []
        self.save_step = save_step
        self.iters = 0
        self.training_time_in_min = 0.0  
        self.training_time_per_epoch_in_min = 0.0
        self.iters_list = []
        self.training_time_in_min = None
        self.training_time_per_itr_in_sec = None
        
        #self.train_dataset = tf.data.Dataset.from_tensor_slices((self.input_trunk_train,self.target_train)).batch(self.batch_size_train)
        
        self.steps_train =  len(self.input_branch_train) #len(self.train_dataset)
        
        if self.validate:
            self.val_dataset = tf.data.Dataset.from_tensor_slices((self.input_trunk_val,self.target_val)).batch(self.batch_size_val)
            self.steps_val =  len(self.val_dataset)
        
        self.beta = 0.0

        #self.bias = tf.Variable(tf.zeros([1,1], dtype=tf_datatype), dtype=tf_datatype)
        
        self.model_trunk = self.model_trunk_net(self.layers,self.lb_trunk,self.ub_trunk)
        
        self.model_branch = self.model_branch_net(self.input_channel,self.num_filters,self.filter_size,self.strides_conv,self.padding,self.strides_pool,self.pool_size,self.FC_layers)
        
        self.len_trunk_w = len(self.model_trunk.trainable_variables)//2
        
        self.ADAM_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_rate)
            

    def model_trunk_net(self,layers,lb_trunk,ub_trunk):
        
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(layers[0]))
        #scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0*(x - lb_trunk)/(ub_trunk - lb_trunk) - 1.0)
        #model.add(scaling_layer)
        for i in range(1,len(layers)-1):
            model.add(tf.keras.layers.Dense(layers[i],activation=tf.keras.activations.get('tanh'), kernel_initializer='glorot_normal'))
        model.add(tf.keras.layers.Dense(layers[-1],use_bias=True,activation=tf.keras.activations.get('linear'), kernel_initializer='glorot_normal'))
        
        return model
    

    def model_branch_net(self,input_channel,num_filters,filter_size,strides_conv,padding,strides_pool,pool_size,FC_layers):

        model = tf.keras.models.Sequential()

        for j in range(len(num_filters)):
            
            if j==0:
                model.add(tf.keras.layers.Conv2D(filters = num_filters[j],kernel_size = filter_size[j], strides = strides_conv, padding = padding, activation='relu', input_shape = input_channel))
            else:
                model.add(tf.keras.layers.Conv2D(filters = num_filters[j],kernel_size = filter_size[j], strides = strides_conv, padding = padding, activation='relu'))

            model.add(tf.keras.layers.MaxPooling2D(pool_size = pool_size, strides = strides_pool))

        model.add(tf.keras.layers.Flatten())

        for i in range(len(FC_layers)-1):
            model.add(tf.keras.layers.Dense(FC_layers[i], activation=tf.keras.activations.get('relu'), kernel_initializer='glorot_normal'))
        model.add(tf.keras.layers.Dense(FC_layers[-1], use_bias=True,activation=tf.keras.activations.get('linear'),
                                        kernel_initializer='glorot_normal'))

        return model
    

    def model_output(self,input_trunk,input_branch):
        
        out_trunk  = self.model_trunk(input_trunk)
        out_branch  =  self.model_branch(input_branch)
        G = out_branch*out_trunk
        output = tf.reduce_sum(G,axis=1,keepdims= True)  #+ self.bias
      
        return output
    
    
    def equation_residual(self,input_trunk,input_branch,k,f):
        
        x, y = input_trunk[:,0:1], input_trunk[:,1:2]
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            u_scaled  = self.model_output(tf.stack([x[:,0],y[:,0]], axis=1),input_branch)
            u = RESCALE(u_scaled,self.mean_target,self.std_target)
            
            u_x = tape.gradient(u,x)
            u_y = tape.gradient(u,y)
            
            k_times_u_x = k*u_x
            k_times_u_y = k*u_y
            
        
        k_times_u_x_x = tape.gradient(k_times_u_x,x)
        k_times_u_y_y = tape.gradient(k_times_u_y,y)
        
        residual = k_times_u_x_x +  k_times_u_y_y + f
        
        del tape
        
        return residual
        
    
    def loss_calculate(self,input_branch,Train_data,BC_data,col_data,k,f):
        
        #input_trunk_data, target_data = Train_data
        
        input_trunk_bc, target_bc = BC_data
        
        input_trunk_col = col_data
        
        #pred_data  = self.model_output(input_trunk_data,input_branch)
        pred_bc  = self.model_output(input_trunk_bc,input_branch)
        
        eqn_residual = self.equation_residual(input_trunk_col,input_branch,k,f)
        
        #loss_data = tf.reduce_mean(tf.square(pred_data - target_data))
        loss_bc = tf.reduce_mean(tf.square(pred_bc - target_bc))
        loss_eqn =  tf.reduce_mean(tf.square(eqn_residual)) 

        #sum_w = self.reguralizer_sum_w()
        #loss_reg = self.beta*sum_w
        
        loss_net = loss_bc + loss_eqn #loss_data + loss_bc + loss_eqn  #+ loss_reg
                          
        #loss_info = (loss, loss_reg)
        
        return loss_net
    
    def loss_validate_calculate(self,input_trunk,input_branch,target):
        
        pred  = self.model_output(input_trunk,input_branch)
        loss = tf.reduce_mean(tf.square(pred - target))

        return loss
    

    def train_variable_list(self):
        
        train_var = []
        train_var.extend(self.model_branch.trainable_variables)
        train_var.extend(self.model_trunk.trainable_variables)
        #train_var.extend([self.bias])
        
        return train_var
 
    
    def get_grad(self,input_branch,Train_data,BC_data,col_data,k,f):
        
        with tf.GradientTape() as tape:
            tape.watch(self.train_variable_list())           
            loss_net = self.loss_calculate(input_branch,Train_data,BC_data,col_data,k,f)
            
        grad_loss_net = tape.gradient(loss_net, self.train_variable_list())
        del tape

        return loss_net, grad_loss_net
    
    def reguralizer_sum_w(self):

        sum_w_trunk = 0
        for i in range(self.len_trunk_w):
            sum_w_trunk = sum_w_trunk + tf.reduce_sum(tf.square(self.model_trunk.trainable_variables[2*i]))
      
        sum_w = sum_w_trunk
        
        return sum_w
    
    
    def loss_calculate_val(self):
        
        val_dataset_iter = iter(self.val_dataset)
        loss_net_avg  = 0.0
        steps = self.steps_val
        for i in range(steps):
            input_trunk, target = next(val_dataset_iter)
            input_branch =  self.input_branch_val[i:i+1]
            loss_net = self.loss_validate_calculate(input_trunk,input_branch,target)
            loss_net_avg = loss_net_avg  + loss_net
            
        loss_net_avg = loss_net_avg/steps

        return loss_net_avg

        
    def train_with_AdamOptimizer(self,iteration):
                
        @tf.function
        def train_step(input_branch,Train_data,BC_data,col_data,k,f):
            loss_net, grad_loss_net =  self.get_grad(input_branch,Train_data,BC_data,col_data,k,f)
            self.ADAM_optimizer.apply_gradients(zip(grad_loss_net, self.train_variable_list()))

            return loss_net

        
        def train_epoch():
            
            #train_dataset_iter = iter(self.train_dataset)
            loss_net_avg  = 0.0
            steps = self.steps_train
            
            for i in range(steps):
                
                #input_trunk_data, target_data  = next(train_dataset_iter)
      
                input_branch =  self.input_branch_train[i:i+1]
                
                col_data = sample_collocation_points(self.col_points)  #input_trunk_data # 
                
                k, f = interpolate_k_f_TF(self.k_train[i].flatten(),self.f_train[i].flatten(),col_data.numpy())
        
                input_trunk_bc, target_bc = sample_boundary_points(self.bc_points,self.mean_target,self.std_target)    
                
                Train_data = ([],[])  #(input_trunk_data,target_data)
                
                BC_data = (input_trunk_bc, target_bc)
                
                loss_net = train_step(input_branch,Train_data,BC_data,col_data,k,f)
                          
                loss_net_avg = loss_net_avg + loss_net
            
            loss_net_avg = loss_net_avg/steps
        
            return loss_net_avg
            
        print('Number of steps in one Epoch : ',self.steps_train)
        
        for itr in range(iteration):
                          
            loss_net_train_ = train_epoch()
            
            if self.iters % self.save_step ==0:
                loss_net_train = loss_net_train_.numpy()
                self.loss_net_train_list.append(loss_net_train)
                print('Epoch : ',self.iters ,' Loss train : ',loss_net_train, end=' ')
            
                if self.validate:
                        loss_net_val_ = self.loss_calculate_val()
                        loss_net_val = loss_net_val_.numpy()         
                        self.loss_net_val_list.append(loss_net_val)
                        print(' Loss val : ', loss_net_val)
                      
            self.iters = self.iters + 1 
            
            
    def train_model(self,iteration):
                          
        if self.iters == 0:
            print('-'*100)
            print("Trunk Network")
            self.model_trunk.summary()
            print('-'*100)
            print("Branch Network")
            self.model_branch.summary()
            print('-'*100)
                          
        print('Adam Optimization Starts')
        time_start = timeR.time()
        self.train_with_AdamOptimizer(iteration)
        time_end = timeR.time()
        time_elapsed = time_end - time_start
        self.training_time_in_min = (time_elapsed)/60.0
        self.training_time_per_epoch_in_min = self.training_time_in_min/iteration
        print('Adam Optimization Ends')
        print("Training Time in min for number of Epochs ",iteration," : ",self.training_time_in_min)
                          
        print('-'*100)
     
    def predict(self,input_trunk,input_branch):
        
        pred = self.model_output(input_trunk,input_branch)
        
        return pred.numpy()
    
    
    


class PI_DeepONet_Darcy_with_Data_sampling_fixed:
    
    def __init__(self,input_trunk_train,input_branch_train,target_train,k_train,f_train,batch_size_train,validate,input_trunk_val,input_branch_val,target_val,batch_size_val,col_points,bc_points,mean_target,std_target,layers,input_channel,num_filters,filter_size,strides_conv,padding,strides_pool,pool_size,FC_layers,lr_rate,save_step):   
        
        self.input_trunk_train = input_trunk_train
        self.input_branch_train = input_branch_train
        self.target_train = target_train
        self.k_train = k_train
        self.f_train = f_train
        self.batch_size_train = batch_size_train
        self.input_trunk_val = input_trunk_val
        self.input_branch_val = input_branch_val
        self.target_val  = target_val
        self.batch_size_val = batch_size_val
        self.col_points = col_points 
        self.bc_points = bc_points
        #self.val_step = val_step
        self.lb_trunk = [] #lb_trunk
        self.ub_trunk = [] #ub_trunk
        self.layers = layers
        self.input_channel = input_channel
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.strides_conv = strides_conv
        self.strides_pool = strides_pool
        self.padding = padding
        self.pool_size = pool_size
        self.FC_layers = FC_layers
        self.lr_rate = lr_rate 
        self.save_step = save_step
        self.validate = validate
        self.mean_target = mean_target
        self.std_target = std_target
        self.loss_net_train_list = []
        self.loss_net_val_list = []
        self.save_step = save_step
        self.iters = 0
        self.training_time_in_min = 0.0  
        self.training_time_per_epoch_in_min = 0.0
        self.iters_list = []
        self.training_time_in_min = None
        self.training_time_per_itr_in_sec = None
        
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.input_trunk_train,self.target_train)).batch(self.batch_size_train)
        
        self.steps_train =  len(self.train_dataset)
        
        self.col_data = sample_collocation_points(self.col_points)
        self.input_trunk_bc, self.target_bc = sample_boundary_points(self.bc_points,self.mean_target,self.std_target)
        
        self.k_list, self.f_list = [], []
        for i in range(self.steps_train):
            k, f = interpolate_k_f_TF(self.k_train[i].flatten(),self.f_train[i].flatten(),self.col_data.numpy())
            self.k_list.append(k)
            self.f_list.append(f)

        if self.validate:
            self.val_dataset = tf.data.Dataset.from_tensor_slices((self.input_trunk_val,self.target_val)).batch(self.batch_size_val)
            self.steps_val =  len(self.val_dataset)
        
        self.beta = 0.0

        #self.bias = tf.Variable(tf.zeros([1,1], dtype=tf_datatype), dtype=tf_datatype)
        
        self.model_trunk = self.model_trunk_net(self.layers,self.lb_trunk,self.ub_trunk)
        
        self.model_branch = self.model_branch_net(self.input_channel,self.num_filters,self.filter_size,self.strides_conv,self.padding,self.strides_pool,self.pool_size,self.FC_layers)
        
        self.len_trunk_w = len(self.model_trunk.trainable_variables)//2
        
        self.ADAM_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_rate)
            

    def model_trunk_net(self,layers,lb_trunk,ub_trunk):
        
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(layers[0]))
        #scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0*(x - lb_trunk)/(ub_trunk - lb_trunk) - 1.0)
        #model.add(scaling_layer)
        for i in range(1,len(layers)-1):
            model.add(tf.keras.layers.Dense(layers[i],activation=tf.keras.activations.get('tanh'), kernel_initializer='glorot_normal'))
        model.add(tf.keras.layers.Dense(layers[-1],use_bias=True,activation=tf.keras.activations.get('linear'), kernel_initializer='glorot_normal'))
        
        return model
    

    def model_branch_net(self,input_channel,num_filters,filter_size,strides_conv,padding,strides_pool,pool_size,FC_layers):

        model = tf.keras.models.Sequential()

        for j in range(len(num_filters)):
            
            if j==0:
                model.add(tf.keras.layers.Conv2D(filters = num_filters[j],kernel_size = filter_size[j], strides = strides_conv, padding = padding, activation='relu', input_shape = input_channel))
            else:
                model.add(tf.keras.layers.Conv2D(filters = num_filters[j],kernel_size = filter_size[j], strides = strides_conv, padding = padding, activation='relu'))

            model.add(tf.keras.layers.MaxPooling2D(pool_size = pool_size, strides = strides_pool))

        model.add(tf.keras.layers.Flatten())

        for i in range(len(FC_layers)-1):
            model.add(tf.keras.layers.Dense(FC_layers[i], activation=tf.keras.activations.get('relu'), kernel_initializer='glorot_normal'))
        model.add(tf.keras.layers.Dense(FC_layers[-1], use_bias=True,activation=tf.keras.activations.get('linear'),
                                        kernel_initializer='glorot_normal'))

        return model
    

    def model_output(self,input_trunk,input_branch):
        
        out_trunk  = self.model_trunk(input_trunk)
        out_branch  =  self.model_branch(input_branch)
        G = out_branch*out_trunk
        output = tf.reduce_sum(G,axis=1,keepdims= True)  #+ self.bias
      
        return output
    
    
    def equation_residual(self,input_trunk,input_branch,k,f):
        
        x, y = input_trunk[:,0:1], input_trunk[:,1:2]
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            u_scaled  = self.model_output(tf.stack([x[:,0],y[:,0]], axis=1),input_branch)
            u = RESCALE(u_scaled,self.mean_target,self.std_target)
            
            u_x = tape.gradient(u,x)
            u_y = tape.gradient(u,y)
            
            k_times_u_x = k*u_x
            k_times_u_y = k*u_y
            
        
        k_times_u_x_x = tape.gradient(k_times_u_x,x)
        k_times_u_y_y = tape.gradient(k_times_u_y,y)
        
        residual = k_times_u_x_x +  k_times_u_y_y + f
        
        del tape
        
        return residual
        
    
    def loss_calculate(self,input_branch,Train_data,BC_data,col_data,k,f):
        
        input_trunk_data, target_data = Train_data
        
        input_trunk_bc, target_bc = BC_data
        
        input_trunk_col = col_data
        
        pred_data  = self.model_output(input_trunk_data,input_branch)
        pred_bc  = self.model_output(input_trunk_bc,input_branch)
        
        eqn_residual = self.equation_residual(input_trunk_col,input_branch,k,f)
        
        loss_data = tf.reduce_mean(tf.square(pred_data - target_data))
        loss_bc = tf.reduce_mean(tf.square(pred_bc - target_bc))
        loss_eqn =  tf.reduce_mean(tf.square(eqn_residual)) 

        #sum_w = self.reguralizer_sum_w()
        #loss_reg = self.beta*sum_w
        
        loss_net = loss_data + loss_bc + loss_eqn  #+ loss_reg
                          
        #loss_info = (loss, loss_reg)
        
        return loss_net
    
    def loss_validate_calculate(self,input_trunk,input_branch,target):
        
        pred  = self.model_output(input_trunk,input_branch)
        loss = tf.reduce_mean(tf.square(pred - target))

        return loss
    

    def train_variable_list(self):
        
        train_var = []
        train_var.extend(self.model_branch.trainable_variables)
        train_var.extend(self.model_trunk.trainable_variables)
        #train_var.extend([self.bias])
        
        return train_var
 
    
    def get_grad(self,input_branch,Train_data,BC_data,col_data,k,f):
        
        with tf.GradientTape() as tape:
            tape.watch(self.train_variable_list())           
            loss_net = self.loss_calculate(input_branch,Train_data,BC_data,col_data,k,f)
            
        grad_loss_net = tape.gradient(loss_net, self.train_variable_list())
        del tape

        return loss_net, grad_loss_net
    
    def reguralizer_sum_w(self):

        sum_w_trunk = 0
        for i in range(self.len_trunk_w):
            sum_w_trunk = sum_w_trunk + tf.reduce_sum(tf.square(self.model_trunk.trainable_variables[2*i]))
      
        sum_w = sum_w_trunk
        
        return sum_w
    
    
    def loss_calculate_val(self):
        
        val_dataset_iter = iter(self.val_dataset)
        loss_net_avg  = 0.0
        steps = self.steps_val
        for i in range(steps):
            input_trunk, target = next(val_dataset_iter)
            input_branch =  self.input_branch_val[i:i+1]
            loss_net = self.loss_validate_calculate(input_trunk,input_branch,target)
            loss_net_avg = loss_net_avg  + loss_net
            
        loss_net_avg = loss_net_avg/steps

        return loss_net_avg

        
    def train_with_AdamOptimizer(self,iteration):
                
        @tf.function
        def train_step(input_branch,Train_data,BC_data,col_data,k,f):
            loss_net, grad_loss_net =  self.get_grad(input_branch,Train_data,BC_data,col_data,k,f)
            self.ADAM_optimizer.apply_gradients(zip(grad_loss_net, self.train_variable_list()))

            return loss_net

        
        def train_epoch():
            
            train_dataset_iter = iter(self.train_dataset)
            loss_net_avg  = 0.0
            steps = self.steps_train
            
            for i in range(steps):
                
                input_trunk_data, target_data  = next(train_dataset_iter)
      
                input_branch =  self.input_branch_train[i:i+1]
                
                #col_data = sample_collocation_points(self.col_points)  #input_trunk_data # 
                
                #k, f = interpolate_k_f_TF(self.k_train[i].flatten(),self.f_train[i].flatten(),col_data.numpy())
            
                k , f = self.k_list[i], self.f_list[i]
                    
                Train_data = (input_trunk_data,target_data)
                BC_data = (self.input_trunk_bc, self.target_bc)
                
                loss_net = train_step(input_branch,Train_data,BC_data,self.col_data,k,f)
                          
                loss_net_avg = loss_net_avg + loss_net
            
            loss_net_avg = loss_net_avg/steps
        
            return loss_net_avg
            
        print('Number of steps in one Epoch : ',self.steps_train)
        
        for itr in range(iteration):
                          
            loss_net_train_ = train_epoch()
            
            if self.iters % self.save_step ==0:
                loss_net_train = loss_net_train_.numpy()
                self.loss_net_train_list.append(loss_net_train)
                print('Epoch : ',self.iters ,' Loss train : ',loss_net_train, end=' ')
            
                if self.validate:
                        loss_net_val_ = self.loss_calculate_val()
                        loss_net_val = loss_net_val_.numpy()         
                        self.loss_net_val_list.append(loss_net_val)
                        print(' Loss val : ', loss_net_val)
                      
            self.iters = self.iters + 1 
            
            
    def train_model(self,iteration):
                          
        if self.iters == 0:
            print('-'*100)
            print("Trunk Network")
            self.model_trunk.summary()
            print('-'*100)
            print("Branch Network")
            self.model_branch.summary()
            print('-'*100)
                          
        print('Adam Optimization Starts')
        time_start = timeR.time()
        self.train_with_AdamOptimizer(iteration)
        time_end = timeR.time()
        time_elapsed = time_end - time_start
        self.training_time_in_min = (time_elapsed)/60.0
        self.training_time_per_epoch_in_min = self.training_time_in_min/iteration
        print('Adam Optimization Ends')
        print("Training Time in min for number of Epochs ",iteration," : ",self.training_time_in_min)
                          
        print('-'*100)
     
    def predict(self,input_trunk,input_branch):
        
        pred = self.model_output(input_trunk,input_branch)
        
        return pred.numpy()
    
    
    



class PI_DeepONet_Darcy_without_Data_sampling_fixed:
    
    def __init__(self,input_trunk_train,input_branch_train,target_train,k_train,f_train,batch_size_train,validate,input_trunk_val,input_branch_val,target_val,batch_size_val,col_points,bc_points,mean_target,std_target,layers,input_channel,num_filters,filter_size,strides_conv,padding,strides_pool,pool_size,FC_layers,lr_rate,save_step):   
        
        self.input_trunk_train = input_trunk_train
        self.input_branch_train = input_branch_train
        self.target_train = target_train
        self.k_train = k_train
        self.f_train = f_train
        self.batch_size_train = batch_size_train
        self.input_trunk_val = input_trunk_val
        self.input_branch_val = input_branch_val
        self.target_val  = target_val
        self.batch_size_val = batch_size_val
        self.col_points = col_points 
        self.bc_points = bc_points
        #self.val_step = val_step
        self.lb_trunk = [] #lb_trunk
        self.ub_trunk = [] #ub_trunk
        self.layers = layers
        self.input_channel = input_channel
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.strides_conv = strides_conv
        self.strides_pool = strides_pool
        self.padding = padding
        self.pool_size = pool_size
        self.FC_layers = FC_layers
        self.lr_rate = lr_rate 
        self.save_step = save_step
        self.validate = validate
        self.mean_target = mean_target
        self.std_target = std_target
        self.loss_net_train_list = []
        self.loss_net_val_list = []
        self.save_step = save_step
        self.iters = 0
        self.training_time_in_min = 0.0  
        self.training_time_per_epoch_in_min = 0.0
        self.iters_list = []
        self.training_time_in_min = None
        self.training_time_per_itr_in_sec = None
        
        #self.train_dataset = tf.data.Dataset.from_tensor_slices((self.input_trunk_train,self.target_train)).batch(self.batch_size_train)
        
        self.steps_train =  len(self.input_branch_train) #len(self.train_dataset)
        
        self.col_data = sample_collocation_points(self.col_points)
        self.input_trunk_bc, self.target_bc = sample_boundary_points(self.bc_points,self.mean_target,self.std_target)
        
        self.k_list, self.f_list = [], []
        for i in range(self.steps_train):
            k, f = interpolate_k_f_TF(self.k_train[i].flatten(),self.f_train[i].flatten(),self.col_data.numpy())
            self.k_list.append(k)
            self.f_list.append(f)

            
        if self.validate:
            self.val_dataset = tf.data.Dataset.from_tensor_slices((self.input_trunk_val,self.target_val)).batch(self.batch_size_val)
            self.steps_val =  len(self.val_dataset)
        
        self.beta = 0.0

        #self.bias = tf.Variable(tf.zeros([1,1], dtype=tf_datatype), dtype=tf_datatype)
        
        self.model_trunk = self.model_trunk_net(self.layers,self.lb_trunk,self.ub_trunk)
        
        self.model_branch = self.model_branch_net(self.input_channel,self.num_filters,self.filter_size,self.strides_conv,self.padding,self.strides_pool,self.pool_size,self.FC_layers)
        
        self.len_trunk_w = len(self.model_trunk.trainable_variables)//2
        
        self.ADAM_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_rate)
            

    def model_trunk_net(self,layers,lb_trunk,ub_trunk):
        
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(layers[0]))
        #scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0*(x - lb_trunk)/(ub_trunk - lb_trunk) - 1.0)
        #model.add(scaling_layer)
        for i in range(1,len(layers)-1):
            model.add(tf.keras.layers.Dense(layers[i],activation=tf.keras.activations.get('tanh'), kernel_initializer='glorot_normal'))
        model.add(tf.keras.layers.Dense(layers[-1],use_bias=True,activation=tf.keras.activations.get('linear'), kernel_initializer='glorot_normal'))
        
        return model
    

    def model_branch_net(self,input_channel,num_filters,filter_size,strides_conv,padding,strides_pool,pool_size,FC_layers):

        model = tf.keras.models.Sequential()

        for j in range(len(num_filters)):
            
            if j==0:
                model.add(tf.keras.layers.Conv2D(filters = num_filters[j],kernel_size = filter_size[j], strides = strides_conv, padding = padding, activation='relu', input_shape = input_channel))
            else:
                model.add(tf.keras.layers.Conv2D(filters = num_filters[j],kernel_size = filter_size[j], strides = strides_conv, padding = padding, activation='relu'))

            model.add(tf.keras.layers.MaxPooling2D(pool_size = pool_size, strides = strides_pool))

        model.add(tf.keras.layers.Flatten())

        for i in range(len(FC_layers)-1):
            model.add(tf.keras.layers.Dense(FC_layers[i], activation=tf.keras.activations.get('relu'), kernel_initializer='glorot_normal'))
        model.add(tf.keras.layers.Dense(FC_layers[-1], use_bias=True,activation=tf.keras.activations.get('linear'),
                                        kernel_initializer='glorot_normal'))

        return model
    

    def model_output(self,input_trunk,input_branch):
        
        out_trunk  = self.model_trunk(input_trunk)
        out_branch  =  self.model_branch(input_branch)
        G = out_branch*out_trunk
        output = tf.reduce_sum(G,axis=1,keepdims= True)  #+ self.bias
      
        return output
    
    
    def equation_residual(self,input_trunk,input_branch,k,f):
        
        x, y = input_trunk[:,0:1], input_trunk[:,1:2]
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            u_scaled  = self.model_output(tf.stack([x[:,0],y[:,0]], axis=1),input_branch)
            u = RESCALE(u_scaled,self.mean_target,self.std_target)
            
            u_x = tape.gradient(u,x)
            u_y = tape.gradient(u,y)
            
            k_times_u_x = k*u_x
            k_times_u_y = k*u_y
            
        
        k_times_u_x_x = tape.gradient(k_times_u_x,x)
        k_times_u_y_y = tape.gradient(k_times_u_y,y)
        
        residual = k_times_u_x_x +  k_times_u_y_y + f
        
        del tape
        
        return residual
        
    
    def loss_calculate(self,input_branch,Train_data,BC_data,col_data,k,f):
        
        #input_trunk_data, target_data = Train_data
        
        input_trunk_bc, target_bc = BC_data
        
        input_trunk_col = col_data
        
        #pred_data  = self.model_output(input_trunk_data,input_branch)
        pred_bc  = self.model_output(input_trunk_bc,input_branch)
        
        eqn_residual = self.equation_residual(input_trunk_col,input_branch,k,f)
        
        #loss_data = tf.reduce_mean(tf.square(pred_data - target_data))
        loss_bc = tf.reduce_mean(tf.square(pred_bc - target_bc))
        loss_eqn =  tf.reduce_mean(tf.square(eqn_residual)) 

        #sum_w = self.reguralizer_sum_w()
        #loss_reg = self.beta*sum_w
        
        loss_net = loss_bc + loss_eqn #loss_data + loss_bc + loss_eqn  #+ loss_reg
                          
        #loss_info = (loss, loss_reg)
        
        return loss_net
    
    def loss_validate_calculate(self,input_trunk,input_branch,target):
        
        pred  = self.model_output(input_trunk,input_branch)
        loss = tf.reduce_mean(tf.square(pred - target))

        return loss
    

    def train_variable_list(self):
        
        train_var = []
        train_var.extend(self.model_branch.trainable_variables)
        train_var.extend(self.model_trunk.trainable_variables)
        #train_var.extend([self.bias])
        
        return train_var
 
    
    def get_grad(self,input_branch,Train_data,BC_data,col_data,k,f):
        
        with tf.GradientTape() as tape:
            tape.watch(self.train_variable_list())           
            loss_net = self.loss_calculate(input_branch,Train_data,BC_data,col_data,k,f)
            
        grad_loss_net = tape.gradient(loss_net, self.train_variable_list())
        del tape

        return loss_net, grad_loss_net
    
    def reguralizer_sum_w(self):

        sum_w_trunk = 0
        for i in range(self.len_trunk_w):
            sum_w_trunk = sum_w_trunk + tf.reduce_sum(tf.square(self.model_trunk.trainable_variables[2*i]))
      
        sum_w = sum_w_trunk
        
        return sum_w
    
    
    def loss_calculate_val(self):
        
        val_dataset_iter = iter(self.val_dataset)
        loss_net_avg  = 0.0
        steps = self.steps_val
        for i in range(steps):
            input_trunk, target = next(val_dataset_iter)
            input_branch =  self.input_branch_val[i:i+1]
            loss_net = self.loss_validate_calculate(input_trunk,input_branch,target)
            loss_net_avg = loss_net_avg  + loss_net
            
        loss_net_avg = loss_net_avg/steps

        return loss_net_avg

        
    def train_with_AdamOptimizer(self,iteration):
                
        @tf.function
        def train_step(input_branch,Train_data,BC_data,col_data,k,f):
            loss_net, grad_loss_net =  self.get_grad(input_branch,Train_data,BC_data,col_data,k,f)
            self.ADAM_optimizer.apply_gradients(zip(grad_loss_net, self.train_variable_list()))

            return loss_net

        
        def train_epoch():
            
            #train_dataset_iter = iter(self.train_dataset)
            loss_net_avg  = 0.0
            steps = self.steps_train
            
            for i in range(steps):
                
                #input_trunk_data, target_data  = next(train_dataset_iter)
      
                input_branch =  self.input_branch_train[i:i+1]
                
                #col_data = sample_collocation_points(self.col_points)  #input_trunk_data # 
                
                #k, f = interpolate_k_f_TF(self.k_train[i].flatten(),self.f_train[i].flatten(),col_data.numpy())
                
                k , f = self.k_list[i], self.f_list[i]
                
                #input_trunk_bc, target_bc = sample_boundary_points(self.bc_points)    
                
                Train_data = ([],[])  #(input_trunk_data,target_data)
                
                BC_data = (self.input_trunk_bc, self.target_bc)
                
                loss_net = train_step(input_branch,Train_data,BC_data,self.col_data,k,f)
                          
                loss_net_avg = loss_net_avg + loss_net
            
            loss_net_avg = loss_net_avg/steps
        
            return loss_net_avg
            
        print('Number of steps in one Epoch : ',self.steps_train)
        
        for itr in range(iteration):
                          
            loss_net_train_ = train_epoch()
            
            if self.iters % self.save_step ==0:
                loss_net_train = loss_net_train_.numpy()
                self.loss_net_train_list.append(loss_net_train)
                print('Epoch : ',self.iters ,' Loss train : ',loss_net_train, end=' ')
            
                if self.validate:
                        loss_net_val_ = self.loss_calculate_val()
                        loss_net_val = loss_net_val_.numpy()         
                        self.loss_net_val_list.append(loss_net_val)
                        print(' Loss val : ', loss_net_val)
                      
            self.iters = self.iters + 1 
            
            
    def train_model(self,iteration):
                          
        if self.iters == 0:
            print('-'*100)
            print("Trunk Network")
            self.model_trunk.summary()
            print('-'*100)
            print("Branch Network")
            self.model_branch.summary()
            print('-'*100)
                          
        print('Adam Optimization Starts')
        time_start = timeR.time()
        self.train_with_AdamOptimizer(iteration)
        time_end = timeR.time()
        time_elapsed = time_end - time_start
        self.training_time_in_min = (time_elapsed)/60.0
        self.training_time_per_epoch_in_min = self.training_time_in_min/iteration
        print('Adam Optimization Ends')
        print("Training Time in min for number of Epochs ",iteration," : ",self.training_time_in_min)
                          
        print('-'*100)
     
    def predict(self,input_trunk,input_branch):
        
        pred = self.model_output(input_trunk,input_branch)
        
        return pred.numpy()