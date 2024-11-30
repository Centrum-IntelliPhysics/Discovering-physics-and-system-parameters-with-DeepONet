clc; close all; clear all;
% number of initial conditions
N_ic = 1;
% number of viscosity conditions
N_visc = 500;
% total number of samples
N = N_ic * N_visc;

% parameters for the Gaussian random field
gamma = 4;
tau = 5;
sigma = 25^2;

% viscosity range
visc_range = linspace(0.01, 0.05, N_visc);

% grid size
s = 4096;
steps = 100;
nn = 101;

input = zeros(N, nn + 1); % +1 to include viscosity
output = zeros(N, steps, nn);

tspan = linspace(0, 1, steps+1);
x = linspace(0, 1, s+1);
X = linspace(0, 1, nn);

sample_index = 1;

u0 = GRF(s/2, 0, gamma, tau, sigma, "periodic");
u0_eval = u0(X);

for j = 1:N_ic
    for v = 1:N_visc
        visc = visc_range(v);
        u = Burgers(u0, tspan, s, visc);
        
        input(sample_index, 1:nn) = u0_eval;
        input(sample_index, end) = visc;
        
        for k = 1:steps+1
            output(sample_index, k, :) = u{k}(X);
        end
        
        sample_index = sample_index + 1;
        if mod(v,100) == 0
            disp(['Completed nu ', num2str(v), ' of ', num2str(N_visc)]);
        end
    end
    
    disp(['Completed initial condition ', num2str(j), ' of ', num2str(N_ic)]);
end

index = randperm(N);

input_train_u0_nu = input(index(1:450), :);
output_train_u = output(index(1:450),:,:);

input_test_u0_nu = input(index(451:500), :);
output_test_u = output(index(451:500),:,:);
% Save the data
save('Burgers_500_nu_20_ic.mat', 'input_train_u0_nu', 'output_train_u', 'input_test_u0_nu', 'output_test_u', 'tspan');