#! /usr/bin/env python
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

# Set up the experiment parameters
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Gaussian kernel density estimation
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data.reshape(-1, 1))
x = np.linspace(-5, 5, 1000)
density = np.exp(kde.score_samples(x.reshape(-1, 1)))

#plot
plt.hist(data, bins=30, density=True, alpha=0.5)
plt.title('Gaussian Kernel Density Estimation(KDE)')
plt.plot(x, density)
plt.show()

# bandwidth different values
bandwidths = [0.1, 0.2, 0.5, 1, 2]

for bandwidth in bandwidths:
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data.reshape(-1, 1))
    density = np.exp(kde.score_samples(x.reshape(-1, 1)))
    plt.plot(x, density, label='Bandwidth={}'.format(bandwidth))

# Create subplots to show best bandwidth value
plt.hist(data, bins=30, density=True, alpha=0.5)
plt.title('Gaussian Kernel Density Estimation (KDE)')
plt.plot(x, density)
plt.legend()
plt.show()
