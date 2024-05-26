# A two-layer neural network-based reduced order Schwarz method for fully nonlinear multiscale elliptic PDEs

The source code for the paper [S. Chen, Z. Ding, Q. Li, S. J. Wright. A reduced order Schwarz method for nonlinear multiscale elliptic equations based on two-layer neural networks. Journal of Computational Mathematics, 42 (2024), pp. 570-596.](https://global-sci.org/intro/article_detail/jcm/22892.html).

## Organization

Most codes are written in Matlab (version R2021a), and the codes to train neural networks are written in Python 3 and Pytorch. 

- `src`: contains the local PDE solvers, the reduced order Schwarz solvers and all the sub routines
- `examples`: contains one example of semilinear elliptic equations and one example of p-Poisson equations

The run time to generate training dataset and the training of neural networks could be between several minutes to several hours depending on the parameters, e.g., the discretization, the dataset size and the number of iterations for training.

## Instructions for use

The instructions for running each case are as follows.

### Semilinear elliptic equations

1. Run semilinear_data.m, which will generate training dataset on each local patch and save them in data_semilinear
2. Run semilinear_init.m, which will generate initial weights used in the training of neural networks
3. Run semilinear_NN.py to train the neural networks that learns the boundary-to-boundary operator in the Schwarz iteration
4. Run semilinear_online.m to solve the semilinear elliptic equations

### p-Poisson equations

1. Run pPoi_data.m, which will generate training dataset on each local patch and save them in data_semilinear
2. Run pPoi_init.m, which will generate initial weights used in the training of neural networks
3. Run pPoi_NN.py to train the neural networks that learns the boundary-to-boundary operator in the Schwarz iteration
4. Run pPoi_online.m to solve the p-Poisson equations

## Cite this work

If you use this code for academic research, you are encouraged to cite the following paper:

```
@article{ChDiLiWr:2021reduced,
  title={A reduced order Schwarz method for nonlinear multiscale elliptic equations based on two-layer neural networks},
  author={Chen, Shi and Ding, Zhiyan and Li, Qin and Wright, Stephen J},
  journal = {Journal of Computational Mathematics},
  year = {2024},
  volume = {42},
  number = {2},
  pages = {570--596}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
