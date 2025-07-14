# Learning Hidden Physics and System Parameters with Deep Operator Networks

Vijay Kag*, [Dibakar Roy Sarkar](https://scholar.google.com/citations?user=Sz4nHdYAAAAJ&hl=en)*, Birupaksha Pal and [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en)

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

## Dataset:

The dataset for all codes on this repository is available [here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/sgoswam4_jh_edu/EvZNjbT37dVFqr3uJxWa-FoByuDaPEMDlvtlTn-8QhlJdw?e=50q2lm).

## Repository Overview:
- **Data generation Folder**: This folder contains scripts that generate example problem data.
- **System Identification Folder**: This folder contains scripts for the example problems.
  - **Burgers**:
    - `training_solution_operator.py`: Implements step 1 of the framework.
    - `DON_inverse_burgers.py`: Implements step 2 of the framework.
    - Wandb/`burgers_inverse_wandb.py`: Example code for hyperparameter tuning.
  - **Reaction Diffusion**:
    - `training_solution_operator.py`: Implements step 1 of the framework.
    - `DON_inverse_reactionDiffusion.py`: Implements step 2 of the framework.
  - **2D Heat L**:
    - `inv_operator_train.ipynb`: Implements step 2 of the framework. 
    - `solution_operator_train.ipynb`: Implements step 2 of the framework.
   
- **Hidden physics Folder**: This folder contains scripts for the example problems.

- **Important Note**: Do not change the `RANDOM_SEED` in the code.


