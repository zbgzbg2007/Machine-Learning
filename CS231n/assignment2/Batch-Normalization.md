# Batch Normalization

BN technique is introduced in this paper:

Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate 
Shift", ICML 2015.

BN layer consists of two parts: one normalization operation (mean subtraction and divided by standard variance) and linear 
transformation (scale and shift).
BN layer is applied after the affine (linear) transformation and before non-linear layer (like ReLU). 
For fully connected network, the normalization is over each feature (that is dimension). So if the input has shape (N, D), 
then the for each of the D features, we do the two operations. For example, for the 1st feature, mean is the average of all N 
values in the first column.
For CNN, the case is a little complicated. As mentioned in the paper:
"we additionally want the normalization to obey the convolutional property – so that different elements of the same feature map, 
at different locations, are normalized in the same way." 
That is, given an output of the convolutional layer with shape (N, C, H, W), we want for each filter (or feature map) all values 
(over all N images and all H\*W  locations) are normalized in the same way. 
I think this is because each filter (or feature map) has its own weights and bias, so its output should be consistent.
Then we need to reshape the array for normalization.

Assume we have input of shape (2, 3, 4) as follows.
```
    array([[[ 0,  1,  2,  3],    
            [ 4,  5,  6,  7],        
            [ 8,  9, 10, 11]],       

           [[12, 13, 14, 15],        
            [16, 17, 18, 19],        
            [20, 21, 22, 23]]]) 
```            
Then we want our new array has shape (C, N\*H\*W) and the first column should be [0, 1, 2, 3, 12, 13, 14, 15]. Then we can pass 
the new array to the BN layer for fully connected network as usual.
To obtain the new array, we need to first transpose and then reshape and then transpose again. Single reshape cannot give us 
the new array since it cannot change the order of the original array.
This is need because we need to apply scale and shift for each filter. For mean and variance, we can simply compute by 
np.mean(a, axis = (0, 2)) and np.var(a, axis = (0, 2)). 

## Expression for propagation

### simple version:
Forward:
```
mu = np.sum(x, axis = 0) / N
xmu = x - mu 
sq = xmu ** 2
var = 1. / N * np.sum(sq, axis = 0) 
std = np.sqrt(var + eps)
inv = 1/sqrt
norm = xmu * inv
out = norm * gamma + beta
```
Backward:
```
dnorm = dout * gamma
dgamma = np.sum(dout * norm, axis = 0)
dbeta = np.sum(dout, axis = 0)
dinv = np.sum(dnorm * xmu, axis = 0)
dxmu1 = dnorm * inv
dstd = dinv * (-std ** (-2))
dvar = dstd * (0.5 * (var + eps) ** (-0.5))
dsq = 1./N * np.ones((N, D)) * dvar
dxmu2 = dsq * 2 * xmu
dxmu = dxmu1 + dxmu2
dx1 = dxmu
dmu = np.sum(-dxmu, axis = 0)
dx2 = 1./N * np.ones((N, D)) * dmu
dx = dx1 + dx2
```
### Another version:
Forward:
```
mu = np.mean(x, axis = 0)
xmu = x - mu
var = np.var(x, axis = 0)
norm = (x - mu) / np.sqrt(var + eps)
out = norm * gamma + beta
```

Backward:
```
dx1 = dnorm/dx * dnorm = dnorm * (var + eps) ** (-0.5)
dnorm_ij/dvar_j = (-0.5) * (x_ij - mu_j) * (var_j + eps) **(-1.5)
dout/dvar_j =  sum_i dout/dnorm_ij * dnorm_ij/dvar_j = sum_i dout_ij * gamma_j * xmu_ij *(-0.5) * (var_j + eps) ** (-1.5) = 
gamma_j * (var_j + eps) ** (-1.5) * (-0.5) * sum_i dout_ij * xmu_ij
dx2 = dout/dvar_j * dvar_j/dx_ij = 1. / N * gamma * (var + eps) ** (-1.5) * np.sum(dout * xmu, axis = 0) * xmu
dnorm_ij/dmu_j = -1 / (var_j + eps) ** (0.5)
dout/dmu_j = sum_i dout/dnorm_ij * dnorm_ij/dmu_j = sum_i dout_ij * gamma_j * (-1) / (var_j + eps) ** (0.5) = 
-gamma_j * (var_j + eps) ** (-0.5) * sum_i dout_ij
dmu_j/dx_ij = 1. / N
dx3 = dout/dmu_j * dmu_j/dx_ij = -1. / N * gamma * (var + eps) ** (-0.5) * np.sum(dout, axis = 0)
dx = dx1 + dx2 + dx3 = 1. / N * gamma * (var + eps) ** (-0.5) * (N * dout - xmu * np.sum(dout * xmu, axis = 0) / (var + eps) - np.sum(dout, axis = 0))
```
