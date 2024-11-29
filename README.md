# Learning Hidden Physics and System Parameters with Deep Operator Networks

Vijay Kag , [Dibakar Roy Sarkar](https://scholar.google.com/citations?user=Sz4nHdYAAAAJ&hl=en), Birupaksha Pal and [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en)

This repository provides two innovative neural operator frameworks for solving challenging scientific problems with sparse observational data:

1. **Hidden Physics Discovery**: Combines DeepONet and physics-informed neural networks to map sparse data to underlying physics, enabling the discovery of governing equations.

![Hidden Physics Framework](https://livejohnshopkins-my.sharepoint.com/personal/droysar1_jh_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fdroysar1%5Fjh%5Fedu%2FDocuments%2FGithub%5Fimage%5Fvideo%2FDON%5Finverse%2FFramework%2Farch%5Fdeeponet%5Fhidden%5Fphysics%5Fpage%2D0001%2Ejpg&parent=%2Fpersonal%2Fdroysar1%5Fjh%5Fedu%2FDocuments%2FGithub%5Fimage%5Fvideo%2FDON%5Finverse%2FFramework&ga=1)

4. **System Parameter Identification**: Uses a pre-trained DeepONet to initialize a physics-constrained inverse model for precise parameter estimation.

### Highlights
- Handles sparse and noisy data while preserving physical consistency.
- Achieves state-of-the-art results on Burgers’ equation and reaction-diffusion systems:
  - **Hidden Physics Discovery**: L2 error of O(10⁻²)
  - **Parameter Identification**: Absolute error of O(10⁻³)

Explore how these frameworks advance scientific discovery with minimal data.

The dataset for all codes on this repository are available [here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/sgoswam4_jh_edu/EvZNjbT37dVFqr3uJxWa-FoByuDaPEMDlvtlTn-8QhlJdw?e=50q2lm).

## For the reference_darcy_PI_DeepONet:
 * PI_DeepONet_Darcy_without_Data_sampling_fixed_RUN.py  => model trained without data apart from BCs (zero value), collocation points are fixed for the sample and do not change while gradient step
 * PI_DeepONet_Darcy_without_Data_sampling_varied_RUN.py => model trained without data apart from BCs (zero value), collocation points are randomly sampled during each gradient step.
 * Do not change the RANDOM_SEED in the codes
