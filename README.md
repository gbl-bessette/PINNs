# Physics-Informed Neural Networks

Analysis of Physics-Informed Neural Networks (PINNs) architectures for solving ODEs. Investigating a non-standard initialization method to accelerate the convergence of unsupervized PINNs: initialize the weights of the 1st layer of the neural network by spanning a large range of values. Inspired by the following paper "Learning in Sinusoidal Spaces with Physics- Informed Neural Networks" by J. C. Wong et. al. (2022).
https://doi.org/10.48550/arXiv.2109.09338

Application to a simple 2D problem sin(x^2+y^2).

Notes:
- Convergence is greatly accelerated for high frequencies by intializing with high weights values. Overfitting issues may occur when initializing with high weights values.
