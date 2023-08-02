# Physics-Informed Neural Networks

Analysis of Physics-Informed Neural Networks (PINNs) architectures for solving ODEs. Investigating a non-standard initialization method to accelerate the convergence of unsupervized PINNs: initialize the weights of the 1st layer of the neural network by spanning a large range of values. Inspired by the following paper "Learning in Sinusoidal Spaces with Physics- Informed Neural Networks" by J. C. Wong et. al. (2022).
https://doi.org/10.48550/arXiv.2109.09338

Application to a simple 2D problem Sin((x-2)^2+y^2) + Sin(x^2+(y-3)^2).

Code can be found under Freq_PINN.py. Results for both supervized and unsupervized learning have been saved in respective folders. 
- Supervized learning: basic configuration where values at every point (BC & collocation) are provided to the PINN as target values.
- Unsupervized learning: target values are only provided on the boundaries of the domain, PINN finds solution inside of the domain on the collocation points, provided only some second order derivatives of the solution.

Each folder contains information on the model, the loss behaviour, as well as some visualization of the results.

Notes:
- Convergence is greatly accelerated for high frequencies by intializing with high weights values. Overfitting issues may occur when initializing with high weights values.
