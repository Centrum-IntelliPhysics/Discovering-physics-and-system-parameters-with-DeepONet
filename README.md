# Learning Hidden Physics and System Parameters with Deep Operator Networks
This repository contains DeepONet for inverse problems and discovering physics.
The dataset for all codes on this repository are available [here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/sgoswam4_jh_edu/EvZNjbT37dVFqr3uJxWa-FoByuDaPEMDlvtlTn-8QhlJdw?e=50q2lm).
## For the reference_darcy_PI_DeepONet:
 * PI_DeepONet_Darcy_without_Data_sampling_fixed_RUN.py  => model trained without data apart from BCs (zero value), collocation points are fixed for the sample and do not change while gradient step
 * PI_DeepONet_Darcy_without_Data_sampling_varied_RUN.py => model trained without data apart from BCs (zero value), collocation points are randomly sampled during each gradient step.
 * Do not change the RANDOM_SEED in the codes
