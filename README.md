# Discovering-physics-with-DeepONet
This repository contains DeepONet for inverse problems and discovering physics.

## For the reference_darcy_PI_DeepONet:
 * PI_DeepONet_Darcy_without_Data_sampling_fixed_RUN.py  => model trained without data apart from BCs (zero value), collocation points are fixed for the sample and do not change while gradient step
 * PI_DeepONet_Darcy_without_Data_sampling_varied_RUN.py => model trained without data apart from BCs (zero value), collocation points are randomly sampled during each gradient step.
 * Do not change the RANDOM_SEED in the codes
