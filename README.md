# Learning Hidden Physics and System Parameters with Deep Operator Networks

[Dibakar Roy Sarkar](https://scholar.google.com/citations?user=Sz4nHdYAAAAJ&hl=en)*, Vijay Kag *, Birupaksha Pal and [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en)

*These authors have made equal contributions to this work.


This repository provides two innovative neural operator frameworks for solving challenging scientific problems with sparse observational data:

1. **Hidden Physics Discovery**: Combines DeepONet and physics-informed neural networks to map sparse data to underlying physics, enabling the discovery of governing equations.

![arch_deeponet_hidden_physics_page-0001](https://github.com/user-attachments/assets/389b5671-6e51-46cc-bce3-a737ddf3dd11)

2. **System Parameter Identification**: This method uses a pre-trained DeepONet to initialize a physics-constrained inverse model for precise parameter estimation.

![Inverse_DON_framework_v3_page-0001](https://github.com/user-attachments/assets/f142d522-fd24-40c8-9e20-4323aea00ac8)


### Highlights
- Handles sparse and noisy data while preserving physical consistency.
- Achieves state-of-the-art results on Burgers’ equation and reaction-diffusion systems:
  - **Hidden Physics Discovery**: L2 error of O(10⁻²)
  - **Parameter Identification**: Absolute error of O(10⁻³)

These results underscore the frameworks’ robustness, efficiency, and potential for solving complex scientific problems with minimal observational data.

## Paper:

The preprint of our paper is available on arXiv: [Learning Hidden Physics and System Parameters with Deep Operator Networks](https://arxiv.org/abs/2412.05133).

## Dataset:

The dataset for all codes on this repository is available [here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/sgoswam4_jh_edu/EvZNjbT37dVFqr3uJxWa-FoByuDaPEMDlvtlTn-8QhlJdw?e=50q2lm).

## Presentation:

A recorded presentation is available [here.](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/sgoswam4_jh_edu/Ele6tiGxnuxFnQe1B6E6g3kB7WHMwPyJs7W5ZHF1Qso_EQ?e=VNTCd1)

## Repository Overview:
- **Data generation Folder**: This folder contains scripts and notebooks (e.g., `helmholtz_data_generation.ipynb`) that generate example problem data.
- **System Identification Folder**: This folder contains scripts for the example problems.
  - **Burgers**:
    - `training_solution_operator.py`: Implements step 1 of the framework.
    - `DON_inverse_burgers.py`: Implements step 2 of the framework.
    - Wandb/`burgers_inverse_wandb.py`: Example code for hyperparameter tuning.
  - **Burgers_dist**:
    - `fwd_sol_operator_train.ipynb`: Implements step 1 of the framework with distributed training.
    - `Inverse_NO.ipynb`: Implements step 2 of the framework with distributed training.
  - **Reaction Diffusion**:
    - `RD_fwd_sol_train.ipynb`: Implements step 1 of the framework.
    - `RD_inverse_op_train.ipynb`: Implements step 2 of the framework.
  - **Reaction_Diffusion_dist**:
    - `RD_fwd_sol_train.ipynb`: Implements step 1 of the framework with distributed training.
    - `RD_inverse_op_train.ipynb`: Implements step 2 of the framework with distributed training.
  - **2D Heat L**:
    - `solution_operator_train.ipynb`: Implements step 1 of the framework.
    - `inv_operator_train.ipynb`: Implements step 2 of the framework.
  - **2D_Heat_L_dist**:
    - `solution_operator_train.ipynb`: Implements step 1 of the framework with distributed training.
    - `inv_operator_train.ipynb`: Implements step 2 of the framework with distributed training.
  - **Helmholtz**:
    - `SID_helmholtz_solution_operator_train.ipynb`: Implements step 1 of the framework.
    - `SID_helmholtz_inv_operator_train.ipynb`: Implements step 2 of the framework.
  - **Helmholtz_dist**:
    - `SID_helmholtz_solution_operator_train.ipynb`: Implements step 1 of the framework with distributed training.
    - `SID_helmholtz_inv_dist.ipynb`: Implements step 2 of the framework with distributed training.
   
- **Hidden physics Folder**: This folder contains scripts for the example problems.

- **Important Note**: Do not change the `RANDOM_SEED` in the code.


