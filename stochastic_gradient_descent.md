# Stochastic gradient descent (SGD)

Let's start from batch gradient descent, where "batch" just means using the entire available samples to perform a single parameter update:

> Repeat for all $j=0,...,n$ { 

>> $\theta_j = \theta_j - \alpha \frac{1}{m} \sum_i (h^i(x^i) - y^i)x_j^i$.

j: dim index, 

i: sample index, 

$\alpha$: learning rate, 

$h^i(x^i) = \sum_j^i (\theta_j x_j^i)$: the linear approximator of y^i.

The last term is the loss function slope $\frac{dL}{d\theta_j}$ averaged over the
entire batch, which is super expensive compared to SGD when we have a large
sample size.


Therefore, SGD reduces the update to:

> Repeat for j=0,...,n {

>> for i=1,...,m {
 
>>> $\theta_j = \theta_j - \alpha (h^i(x^i) - y^i)x_j^i$, 

by randomly shuffle dataset (reduce correlation between updates) to speed up the optimization convergence rate.

# Reference
[stochastic gradient descent by Andrew Ng](https://www.youtube.com/watch?v=W9iWNJNFzQI)
