"""Inverse_500_nu_data

# Viscous Burgers Equation (Forward mode auto diff)

## Maping viscosity space to the solution space

$$ \frac{ds}{dt} + s \frac{ds}{dx} = \nu \frac{d^2s}{dx^2}$$

$$ s(x,0) = u(x), x \in (0,1),$$

$$ s(0,t) = s(1,t), $$

$$ \frac{ds(0,t)}{dx} = \frac{ds(1,t)}{dx},$$
"""


import wandb
import numpy as np
from types import SimpleNamespace
import random

wandb.login(key='fb304ee258f187fc45819f25e0b75084cef8870f') # Please use your own key


from flax import linen as nn
from typing import Sequence, Tuple
import jax
import jax.numpy as jnp
from torch.utils import data
from functools import partial
import time
import optax
import scipy.io
import os
import argparse
import shutil
import pandas as pd
from jax import jvp
import pickle
from sklearn import metrics
from termcolor import colored
import random
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""# Inputs"""

## All inputs

path = os.path.join(os.getcwd(), '/home/droysar1/data_sgoswam4/DeepONet_inverse/Burgers/Burgers_500_nu_20_ic.mat')  # Please use the matlab script to generate data

# You need to define a config file in the form of dictionary or yaml
sweep_config = {
    'method': 'bayes',
    'name' : 'DON_inv_t2',
    'metric': {
      'name': 'l2_err_test',
      'goal': 'minimize'
    },
    'parameters': {
         'hidden_size':{
            'values':[3, 4, 5]
        },
        'neuron_nu':{
            'values':[32,64,128,256]
        },
        'activation': {
            'values': ['sin','tanh','relu']
        }

    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='DON_inverse_burgers_trail')


def main():
    '''
    WandB calls main function each time with differnet combination.

    We can retrive the same and use the same values for our hypermeters.

    '''

    with wandb.init() as run:

        run_name="hs"+str(wandb.config.hidden_size)+"-nnum_n"+str(wandb.config.neuron_nu)
        wandb.run.name=run_name
        
        n_train = 8500
        p_data_train = 300
        p_res_train = 2500
        batch_size = 2500
        n_test = 101
        n_sensors = p_data_train #101*101
        branch_layers = [64, 64, 64]
        
        branch_nu_layers = [wandb.config.neuron_nu] * wandb.config.hidden_size
        
        branch_input_features = 1
        trunk_layers = [64, 64, 64]
        trunk_input_features = 2
        hidden_dim = 100
        hidden_dim_nu = 1
        p_test = 101
        result_dir = './'
        epochs = 80000
        vis_iter = 1000
        lr = 1e-3
        transition_steps = 2000
        decay_rate = 0.9
        
        data_ref = scipy.io.loadmat(path)
        u_sol_train = jnp.array(data_ref['output_train_u'])
        
        ic_and_nu_train = jnp.array(data_ref['input_train_u0_nu'])
        
        nu_train = ic_and_nu_train[:,-1]
        # nu_train = ic_and_nu_train[:,-1]
        
        u_sol_test = jnp.array(data_ref['output_test_u'])
        
        ic_and_nu_test = jnp.array(data_ref['input_test_u0_nu'])
        nu_test = ic_and_nu_test[:,-1]
        # nu_test = ic_and_nu_test[:,-1]
        
        print("output_train: ", str(u_sol_train.shape))   # (n_train, nt, nx)
        print("nu_train: ", str(nu_train.shape))  # (n_train,)
        print("output_test: ", str(u_sol_test.shape))  # (n_test, nt, nx)
        print("nu_test: ", str(nu_test.shape)) # (n_test,)
        
        def extract_points(data, num_points=300):
            seed = 0
            key = jax.random.PRNGKey(seed)
            samples, ntt, nxx = data.shape
            flattened_data = data.reshape(samples, ntt * nxx)
            random_indices = jax.random.choice(key, ntt * nxx, shape=(num_points,), replace=False)
            t_indices = random_indices // ntt
            x_indices = random_indices % nxx
            t = t_indices / (ntt - 1)
            x = x_indices / (nxx - 1)
        
            # Create y array with (t, x) coordinates
            location = jnp.stack([t, x], axis=-1)
        
            # Create a grid representing the original domain
            tt = np.linspace(0, 1, ntt)
            xx = np.linspace(0, 1, nxx)
            XX, TT = jnp.meshgrid(xx, tt, indexing='ij')
        
            y_test = jnp.stack([TT.flatten(), XX.flatten()], axis=-1)
        
            # Sample s values from u_data
            extracted_data = data[:,t_indices, x_indices]
        
            return extracted_data, location, y_test
        
        # Extract 300 points from each sample
        data_output, location_test, y_test = extract_points(u_sol_test, p_data_train)
        
        """# Data generator class"""
        
        # Data Generator
        class DataGenerator(data.Dataset):
            def __init__(self, u, y, s, nu, ics, batch_size, gen_key):
                self.u = u
                self.y = y
                self.s = s
                self.nu = nu
                self.ics = ics
                self.N = nu.shape[0]
                self.batch_size = batch_size
                self.key = gen_key
        
            def __getitem__(self, index):
                """Generate one batch of data"""
                self.key, subkey = jax.random.split(self.key)
                inputs, outputs, nu, ics = self.__data_generation(subkey)
                return inputs, outputs, nu, ics
        
            @partial(jax.jit, static_argnums=(0,))
            def __data_generation(self, key_i):
                """Generates data containing batch_size samples"""
                idx = jax.random.choice(key_i, self.N, (self.batch_size,), replace=False)
                s = self.s[idx, :]
                y = self.y[:, :]
                u = self.u[idx, :]
                nu = self.nu[idx]
                ics = self.ics[idx]
                # Construct batch
                inputs = (u, y)
                outputs = s
                return inputs, outputs, nu, ics
        
        """# Data generation for each samples"""
        
        def generate_random_data_points(nu_train_i, u_sol_train_i, key, num_points=300):
            
            # Generate random indices for sampling points
            num_t, num_x = u_sol_train_i.shape
            total_points = num_t * num_x
            random_indices = jax.random.choice(key, total_points, shape=(num_points,), replace=False)
            
            # Calculate t and x coordinates for the sampled points
            t_indices = random_indices // num_x
            x_indices = random_indices % num_x
            t = t_indices / (num_t - 1)
            x = x_indices / (num_x - 1)
            
            # Create y array with (t, x) coordinates
            y = jnp.stack([t, x], axis=-1)
            
            # Sample s values from u_data
            u = u_sol_train_i[t_indices, x_indices]
            
            # s = u_sol_train_i
            
            return u, y, u, nu_train_i
        
        def generate_one_res_training_data(nu_train_i, u_sol_train_i, key, num_points=300, p=2500):
        
            num_t, num_x = u_sol_train_i.shape
            total_points = num_t * num_x
            random_indices = jax.random.choice(key, total_points, shape=(num_points,), replace=False)
        
            # Calculate t and x coordinates for the sampled points
            t_indices = random_indices // num_t
            x_indices = random_indices % num_x
            t = t_indices / (num_t - 1)
            x = x_indices / (num_x - 1)
        
            # Create y array with (t, x) coordinates
            p_grid = int(jnp.sqrt(p))  # sqrt to get square grid
        
            t_res_i = jax.random.uniform(key, shape=(p_grid,), minval=0, maxval=1)
            t_res_i = jnp.sort(t_res_i)  # Sort to maintain increasing order
            x_res_i = jax.random.uniform(key, shape=(p_grid,), minval=0, maxval=1)
            x_res_i = jnp.sort(x_res_i)  # Sort to maintain increasing order
            x_res, t_res = jnp.meshgrid(x_res_i, t_res_i, indexing='ij')
        
            y = jnp.stack([t_res.flatten(), x_res.flatten()], axis=-1)  # Shape: (p, 2)
        
            # Sample s values from u_data
            u = u_sol_train_i[t_indices, x_indices]
        
            s = u_sol_train_i
        
            return u, y, s, nu_train_i
        
        
        
        # Create test data
        seed = 0
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 6)
        
        # known u data
        u_data_train, y_data_train, s_data_train, nu_data_train = (jax.vmap(generate_random_data_points,
                                                             in_axes=(0, 0, None, None))
                                                    (nu_train, u_sol_train, key, p_data_train))
        
        print("u_data_train: ", str(u_data_train.shape))  # Expected: (N_train, 300)
        print("s_data_train: ", str(s_data_train.shape))  # Expected: (N_train, 101, 101)
        print("y_data_train: ", str(y_data_train.shape))  # Expected: (N_train, 300, 2)
        print("nu_data_train: ", str(nu_data_train.shape))  # Expected: (N_train,)
        
        # Residual data
        # key_res = jax.random.split(keys[0], n_train)
        u_res_train, y_res_train, s_res_train, nu_res_train = (jax.vmap(generate_one_res_training_data,
                                                         in_axes=(0, 0, None, None, None))
                                                 (nu_train, u_sol_train, key, p_data_train, p_res_train))
        
        print("u_res_train: ", str(u_res_train.shape)) # Expected: (N_train, 300)
        print("s_res_train: ", str(s_res_train.shape))  # Expected: (N_train, 101, 101)
        print("y_res_train: ", str(y_res_train.shape)) # Expected: (N_train, 2500, 2)
        print("nu_res_train: ", str(nu_res_train.shape)) # Expected: (N_train, )
        
        
        
        # Create data generators
        data_dataset = DataGenerator(u_data_train, y_data_train, s_data_train, nu_data_train, ic_and_nu_train[:, :-1], batch_size, keys[3])
        res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, nu_res_train, ic_and_nu_train[:, :-1], batch_size, keys[2])
        
        """# DeepONet class"""
        
        # DeepONet class
        
        class DeepONet(nn.Module):
            branch_layers: Sequence[int]
            trunk_layers: Sequence[int]
            output_dim: int = 1
        
            @nn.compact
            def __call__(self, branch_x, trunk_x):
        
                init = nn.initializers.glorot_normal()
        
                if branch_x.ndim == 1:
                    branch_x = jnp.expand_dims(branch_x, axis=-1)
        
        #         print("branch_x_in: ", str(branch_x.shape))
                # Branch network [batch_size, 1]
                for i, fs in enumerate(self.branch_layers[:-1]):
                    branch_x = nn.Dense(fs, kernel_init=init, name=f"branch_{i}")(branch_x)
                    branch_x = nn.activation.tanh(branch_x)
                branch_x = nn.Dense(self.branch_layers[-1], name=f"branch_{i+1}", kernel_init=init)(branch_x)
                #[batch_size, p]
        #         print("branch_x_out: ", str(branch_x.shape))
        
                # Trunk network
                # [x_num, 2]
                # Ensure trunk_x has at least 2 dimensions
                if trunk_x.ndim == 1:
                    trunk_x = jnp.expand_dims(trunk_x, axis=0)
        
        #         print("trunk_x_in: ", str(trunk_x.shape))
        
                for i, fs in enumerate(self.trunk_layers):
                    trunk_x = nn.Dense(fs, kernel_init=init, name=f"trunk_{i}")(trunk_x)
                    trunk_x = nn.activation.tanh(trunk_x)
        
        #         print("trunk_x_out: ", str(trunk_x.shape))
        
                 # [x_num, p]
        
                # Compute the final output
                # Input shapes:
                # branch: [batch_size, p]
                # trunk: [x_num, p]
                # output: [batch_size, x_num]
                result = jnp.einsum('ij,kj->ik', branch_x, trunk_x)
        
                # Add bias
                bias = self.param('output_bias', nn.initializers.zeros, (self.output_dim,))
                result += bias
        
                return result
        
        """# DeepONet for nu"""
        
        # NuNet class
        
        class NuNet(nn.Module):
            branch_nu_layers: Sequence[int]
            output_dim: int = 1
        
            @nn.compact
            def __call__(self, branch_nu_x):
        
                init = nn.initializers.glorot_normal()
        
                if branch_nu_x.ndim == 1:
                    branch_nu_x = jnp.expand_dims(branch_nu_x, axis=-1)
        
        #         print("branch_x_in: ", str(branch_x.shape))
                # Branch network [batch_size, 1]
                for i, fs in enumerate(self.branch_nu_layers[:-1]):
                    branch_nu_x = nn.Dense(fs, kernel_init=init, name=f"branch_nu_{i}")(branch_nu_x)
                    if wandb.config.activation == 'relu':
                        branch_nu_x = nn.activation.relu(branch_nu_x)
                    elif wandb.config.activation == 'tanh':
                        branch_nu_x = nn.activation.tanh(branch_nu_x)
                    elif wandb.config.activation == 'sin':
                        branch_nu_x = jnp.sin(branch_nu_x)
                branch_nu_x = nn.Dense(self.branch_nu_layers[-1], name=f"branch_nu_{i+1}", kernel_init=init)(branch_nu_x)
                #[batch_size, p]
        #         print("branch_nu_x: ", str(branch_nu_x.shape))
        
                result = jnp.sum(branch_nu_x, axis=1)
                # output: [batch_size,]
        #         print("result: ", str(result.shape))
        
                # Add bias
                bias = self.param('output_bias', nn.initializers.zeros, (self.output_dim,))
                result += bias
        
                return result
        
        
        """# Helper functions"""
        
        @partial(jax.jit)
        def mse(y_true, y_pred):
            return jnp.mean(jnp.square(y_true - y_pred))
        
        @partial(jax.jit)
        def mse_single(y_pred):
            return jnp.mean(jnp.square(y_pred))
        
        @partial(jax.jit, static_argnums=(0,))
        def apply_net(model_fn, params, branch_input, *trunk_in):
            # Define forward pass for normal DeepOnet that takes series of trunk inputs and stacks them
            if len(trunk_in) == 1:
                trunk_input = trunk_in[0]
            else:
                trunk_input = jnp.stack(trunk_in, axis=-1)
            out = model_fn(params, branch_input, trunk_input)
        
            # Reshape to vector for single output for easier gradient computation
            # TODO: Adapt / Check squeeze
            if out.shape[1]==1:
                out = jnp.squeeze(out, axis=1)
            return out
        
        @partial(jax.jit, static_argnums=(0,))
        def apply_net_nu(model_fn_nu, params_nu, branch_input_data):
            out = model_fn_nu(params_nu, branch_input_data)
        
            # TODO: Adapt / Check squeeze
            if out.shape[0]==1:
                out = jnp.squeeze(out, axis=0)
            return out
        
        
        @partial(jax.jit, static_argnums=(0, 1, 2, 3))
        def step(optimizer, loss_fn, model_fn_nu, model_fn, opt_state, params_step, res_batch, data_batch):
            loss, gradient = jax.value_and_grad(loss_fn, argnums=2)(model_fn_nu, model_fn, params_step, res_batch, data_batch)
            updates, opt_state = optimizer.update(gradient, opt_state)
            params_step = optax.apply_updates(params_step, updates)
        
            return loss, params_step, opt_state
        
        """# set the model pretrained model"""
        
        # Initialize model and params
        # make sure trunk_layers and branch_layers are lists
        trunk_layers = [trunk_layers] if isinstance(trunk_layers, int) else trunk_layers
        
        branch_layers = [branch_layers] if isinstance(branch_layers, int) else branch_layers
        
        # add output features to trunk and branch layers
        trunk_layers = trunk_layers + [hidden_dim]
        branch_layers = branch_layers + [hidden_dim]
        
        # Convert list to tuples
        trunk_layers = tuple(trunk_layers)
        branch_layers = tuple(branch_layers)
        
        num_outputs = 1
        model = DeepONet(branch_layers, trunk_layers, num_outputs)
        
        # model function
        model_fn = jax.jit(model.apply)
        
        """# model save function"""
        
        def save_model_params(params, result_dir, filename='model_params.pkl'):
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
        
            save_path = os.path.join(result_dir, filename)
            with open(save_path, 'wb') as f:
                pickle.dump(params, f)
        
        def load_model_params(result_dir):
            load_path = os.path.join(result_dir)
            with open(load_path, 'rb') as f:
                params = pickle.load(f)
            return params
        
        save = True
        params_loaded = load_model_params('/home/droysar1/data_sgoswam4/DeepONet_inverse/Burgers/model_params_best_v1.pkl')

        print("Loaded best model parameters")
        
        """# Test error function"""
        
        def loss_test_l2_error(model_fn, params, data_output, u_sol_test, return_data=False):
        
            def ff_net(t_test, x_test):
                s_pred = apply_net(model_fn, params, data_output, t, x)
                return s_pred
        
            # print(u_sol_test.shape)
            # s_pred = ff_net(t, x)
             # Generate random indices for sampling points
            _, num_t, num_x = u_sol_test.shape
            total_points = num_t * num_x
            random_indices = jax.random.choice(key, total_points, shape=(300,), replace=False)
            
            # Calculate t and x coordinates for the sampled points
            t_indices = random_indices // num_x
            x_indices = random_indices % num_x
            t = t_indices / (num_t - 1)
            x = x_indices / (num_x - 1)
            
            # Create y array with (t, x) coordinates
            y = jnp.stack([t, x], axis=-1)
            
            # Sample s values from u_data
            u = u_sol_test[:, t_indices, x_indices]
        
            s_pred = ff_net(t, x)
            # s_pred = s_pred.reshape(u_sol_test.shape[0], u_sol_test.shape[1], u_sol_test.shape[2])
        
            loss_test = mse(u, s_pred)
        
            def l2_norm_error(a, b):
                """Calculate L2 norm error between two 2D arrays."""
                return jnp.linalg.norm(a - b)/jnp.linalg.norm(a)
        
            l2_error = jnp.mean(jax.vmap(l2_norm_error)(u, s_pred))
        
            if return_data:
                return loss_test, l2_error, s_pred
            else:
                return loss_test, l2_error
        
        """# Test visualization"""
        
        def loss_test_l2_error_one_sample(model_fn, params, data_output, u_sol_test, idx, return_data=False):
        
            t_test = jnp.linspace(0, 1, u_sol_test.shape[1])
            x_test = jnp.linspace(0, 1, u_sol_test.shape[2])
        
            xx_test, tt_test = jnp.meshgrid(x_test, t_test, indexing='ij')
        
            y = jnp.stack([tt_test.flatten(), xx_test.flatten()], axis=-1) #(100*100, 2)
        
            t = y[:,0]
            x = y[:,1]
        
            data_output_i = data_output[idx, :]
            data_output_i = data_output_i[jnp.newaxis,:]
        
            def ff_net(t_test, x_test):
                s_pred = apply_net(model_fn, params, data_output_i, t, x)
                return s_pred
        
            s_pred = ff_net(t, x)
            s_pred = s_pred.reshape(u_sol_test.shape[1], u_sol_test.shape[2])
        
            loss_test = mse(u_sol_test[idx, :, :], s_pred)
        
            def l2_norm_error(a, b):
                """Calculate L2 norm error between two 2D arrays."""
                return jnp.linalg.norm(a - b)/jnp.linalg.norm(a)
        
            l2_error = jnp.mean(l2_norm_error(u_sol_test[idx,:,:], s_pred))
        
        
            if return_data:
                return loss_test, l2_error, s_pred, u_sol_test[idx,:,:]
            else:
                return loss_test, l2_error
        
        def visualize_show(model_fn, params, result_dir, epoch, data_output, u_sol_test, nu_test, idx, test=False):
            # Generate data, and obtain error
            loss_test, l2_error, s_pred, s_test = loss_test_l2_error_one_sample(model_fn, params, data_output, u_sol_test, idx, return_data=True)
        
            s_test = s_test.T
            
            x = jnp.linspace(0, 1, s_pred.shape[1])
            t = jnp.linspace(0, 1, s_pred.shape[0])
        
            x_test, t_test = jnp.meshgrid(x, t, indexing='ij')
        
        
            # Calculate R² score
            r2_value = metrics.r2_score(s_test.flatten(), s_pred.flatten())
            r2_value = float('%.4f' % r2_value)
        
            # Plot
            fig = plt.figure(figsize=(12, 4))
        
            # Adjusting layout for more vertical space
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.7, wspace=0.4, hspace=0.1)
        
            # Plot Exact u over time using pcolor
            ax = fig.add_subplot(1, 3, 1)
            plt.pcolor(x_test, t_test, s_test, cmap='jet', shading='auto')
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=14)
            ax.set_xlabel(r'$x$', fontsize=14)
            ax.set_ylabel(r'$t$', fontsize=14)
            ax.set_title('True field', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=14)
            plt.tight_layout()
        
            # Plot Predicted u over time using pcolor
            ax = fig.add_subplot(1, 3, 2)
            plt.pcolor(x_test, t_test, s_pred, cmap='jet', shading='auto')
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=14)
            ax.set_xlabel(r'$x$', fontsize=14)
            ax.set_ylabel(r'$t$', fontsize=14)
            ax.set_title('Predicted field', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=14)
        
            plt.tight_layout()
        
            # Plot Absolute error using pcolor
        
            u_diff = s_test - s_pred
        
            ax = fig.add_subplot(1, 3, 3)
            plt.pcolor(x_test, t_test, jnp.abs(u_diff), cmap='jet', shading='auto')
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=14)
            ax.set_xlabel(r'$x$', fontsize=14)
            ax.set_ylabel(r'$t$', fontsize=14)
            ax.set_title('Absolute error', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=14)
        
            plt.tight_layout()
        
        
            if test:
                plt.suptitle(f'Test L2: {l2_error:.3e}, R2: {r2_value}, nu: {nu_test[idx]:.3e}', y=1.05, fontsize=14)
            else:
                plt.suptitle(f'Train L2: {l2_error:.3e}, R2: {r2_value:.3e}', y=1.05, fontsize=14)
        
            # Show or save the plot
        #     if save:
        #         plt.savefig(os.path.join(result_dir, f'Test_Sample_{idx}.pdf'))
            plt.show()
            plt.close()
            print(colored('#' * 230, 'green'))
            return r2_value  # Return R² score for each visualization
        
        """# Loss function"""
        
        def loss_res(model_fn_nu, model_fn, params_nu, params_loaded, res_batch):
            # Fetch data
            inputs, sol, nuu,_ = res_batch
            u_data, y_res = inputs
        
            rand_num = random.randint(0, n_train)
            t = y_res[rand_num, :, 0]
            x = y_res[rand_num, :, 1]
        
            def f(t, x):
                return apply_net(model_fn, params_loaded, u_data, t, x)
        
        
            # Compute forward pass
            s = f(t, x)
        
            nu_pred = apply_net_nu(model_fn_nu, params_nu, u_data)
        
            _, s_x = jax.jvp(lambda x: f(t, x), (x,), (jnp.ones_like(x),)) # changed f(t, x) to f(x, t)
            _, s_t = jax.jvp(lambda t: f(t, x), (t,), (jnp.ones_like(t),))
        
        
            _, s_xx = jax.jvp(lambda x: jax.jvp(lambda x: f(t, x), (x,), (jnp.ones_like(x),))[1], (x,), (jnp.ones_like(x),))
        
        
            nu_pred = nu_pred.reshape(-1,1)
            nuu = nuu.reshape(-1,1)
        
            # Compute residual
            pred = s_t + s * s_x - nu_pred * s_xx
        
            # Compute loss
            loss = mse_single(pred)
            return loss
        
        def loss_data(model_fn, params, data_batch):
        
            inputs, outputs, _,_ = data_batch
            u_data, y = inputs
        
            t = y[1, :, 0] # pick y for one sample (y is same for all samples)
            x = y[1, :, 1]
        
            s_pred = apply_net(model_fn, params, u_data, t, x)
            s_pred = s_pred.reshape(outputs.shape[0], outputs.shape[1])
        
            loss = mse(outputs, s_pred)
            return loss
        
        def loss_fn(model_fn_nu, model_fn, params, res_batch, data_batch):
            params_loaded, params_nu = params
            loss_res_i = loss_res(model_fn_nu, model_fn, params_nu, params_loaded, res_batch)
            loss_data_i = loss_data(model_fn, params_loaded, data_batch)
        
            loss_value =  1.0 * loss_res_i  + 1*loss_data_i
        
            return loss_value
        
        """# Initialize model"""
        
        # branch_nu_layers = [64, 64, 64]
        
        """# set the model"""
        
        # Initialize model and params
        
        branch_nu_layers = [branch_nu_layers] if isinstance(branch_nu_layers, int) else branch_nu_layers
        
        # add output features to trunk and branch layers
        branch_nu_layers = branch_nu_layers + [hidden_dim_nu]
        
        # Convert list to tuples
        branch_nu_layers = tuple(branch_nu_layers)
        
        num_outputs = 1
        model_nu = NuNet(branch_nu_layers, num_outputs)
        
        params_nu = model_nu.init(key, jnp.ones(shape=(1, n_sensors * branch_input_features)))
        
        # Print model from parameters
        print('--- model_summary ---')
        # count total params
        total_params_nu = sum(x.size for x in jax.tree_util.tree_leaves(params_nu))
        print(f'total params nu: {total_params_nu}')
        print('--- model_summary ---')
        
        # model function
        model_fn_nu = jax.jit(model_nu.apply)
        
        
        """# Test error function"""
        
        def loss_test_l2_error_nu(model_fn_nu, params_nu, data_output, u_sol_test, nu_test, return_data=False):
        
            nu_pred = apply_net_nu(model_fn_nu, params_nu, u_sol_test)
        
            loss_test = mse(nu_test, nu_pred)
        
            def rel_abs_error(a, b):
                """Calculate realtive absolute error"""
                return jnp.abs(a - b)/a
        
            l2_error = jnp.mean(jax.vmap(rel_abs_error)(nu_test, nu_pred))
        
            if return_data:
                return loss_test, l2_error, nu_pred
            else:
                return loss_test, l2_error
        
        """# Define optimiser  of nu prediction"""
        
        # Define optimizer with optax (ADAM)
        # optimizer
        
        lr_scheduler = optax.exponential_decay(lr, transition_steps, decay_rate)
        
        # lr_scheduler = optax.constant_schedule(1e-3)
        
        optimizer = optax.adam(learning_rate=lr_scheduler)
        
        params = (params_loaded, params_nu) # include both parameters for training
        opt_state = optimizer.init(params)
        
        # Data
        data_data = iter(data_dataset)
        res_data = iter(res_dataset)
        
        
        """# Training loop"""
        
        # Iterations
        epochs = epochs
        log_iter = 1000

        for it in range(epochs):
            if it == 1:
                # start timer and exclude first iteration (compile time)
                start = time.time()
            # Fetch data
            res_batch = next(res_data)
            data_batch = next(data_data)
        
            # Do Step
        
            loss, params, opt_state = step(optimizer, loss_fn, model_fn_nu, model_fn, opt_state,
                                           params, res_batch, data_batch)
            if it % log_iter == 0:
                # Compute losses
                # def loss_res(model_fn_nu, model_fn, params_nu, params_loaded, res_batch):
                params_loaded, params_nu = params
                loss = loss_fn(model_fn_nu, model_fn, params, res_batch, data_batch)
                loss_res_value = loss_res(model_fn_nu, model_fn, params_nu, params_loaded, res_batch)
                loss_data_value = loss_data(model_fn, params_loaded, data_batch)
        
                loss_test, l2_err_test = loss_test_l2_error(model_fn, params_loaded, data_output, u_sol_test, return_data=False)

                wandb.log({'loss': loss})
                wandb.log({'loss_res_value': loss_res_value})
                wandb.log({'loss_data_value': loss_data_value})
                wandb.log({'l2_err_test': l2_err_test})
                wandb.log({'loss_test': loss_test})


wandb.agent(sweep_id, function=main,count=50) # calls main function for count number of times.
wandb.finish()