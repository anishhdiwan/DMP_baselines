This repository contains baseline implementations of several methods to better understand amd compare with a proposed diffusion cost functions method. 

It assumes that [IsaacGym Preview_v4](https://developer.nvidia.com/isaac-gym) and [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) are both installed (preferably using the conda env). NOTE: custom envs can also be created using the isaacgym api, however, at present baselines are tested on Cartpole as defined in IsaacGymEnvs.

# Cross Entropy Method (for optimization)

Kobilarov M. [Cross-entropy motion planning](https://doi.org/10.1177/0278364912444543). The International Journal of Robotics Research. 2012 Jun;31(7):855-71.

Botev ZI, Kroese DP, Rubinstein RY, L’Ecuyer P. [The cross-entropy method for optimization](https://www.sciencedirect.com/science/article/pii/B9780444538598000035). InHandbook of statistics 2013 Jan 1 (Vol. 31, pp. 35-59). Elsevier.



## CEM Algorithm


1. Assuming that actions are conditioned on the current state and are normally distributed, choose initial parameters $\mu^{(0)}$ and $\sigma^{(0)};$ set $t$ = 1

2. Sample `N ` actions $X_1, X_2, ..., X_n$ from Gaussian distribution with mean and variance $\mu^{(t)}, \sigma^{(t)}$

3. Select the best `Ne` samples to update $\mu^{(t)}, \sigma^{(t)}$ (this can also be done recursively)

4. Stop if convergence criteria are satisfied; otherwise, increase $t$ by 1 and repeat from step 2.



## CEM Implementation w/ IsaacGym

To run the isaacgym version of cem for cartpole execute `python cem_cartpole.py` within the rlgpu conda env.
