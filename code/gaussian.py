import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
import subprocess
from sklearn.mixture import GaussianMixture

os.makedirs('./output/gaussian', exist_ok=True)
sns.set(color_codes=True)
sns.set(font_scale=1.5)

plt.figure(figsize=(10, 8))
data = np.linspace(-2.5, 2.5, 1000)
mean = 0.0
std = np.sqrt(.5)
dist = (data - mean) ** 2
y = (1 / (2 * (std ** 2))) * np.exp(-(1 / (2 * (std **2))) * dist)
plt.plot(data, y)
plt.tight_layout()
plt.ylabel('')
plt.savefig('./output/gaussian/gaussian1.png')


data = np.linspace(-2.5, 2.5, 1000)
stds = [.0125, .025, .05, .1, .2, .5][::-1]
stds = [np.sqrt(x) for x in stds]
legends = []
for i, std in enumerate(stds):
    plt.figure(figsize=(10, 8))
    for std in stds[:i+1]:
        mean = 0.0
        dist = (data - mean) ** 2
        y = (1 / (2 * np.pi * (std ** 2))) * np.exp(-(dist / (2 * (std **2))))
        plt.plot(data, y, label = f'beta = {1 / (2 * (std ** 2)):.2f}')
    plt.ylim([0, 8])
    plt.legend()

    plt.tight_layout()
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(f'./output/gaussian/beta_var_{i}.png')