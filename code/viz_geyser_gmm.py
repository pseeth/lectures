import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
import subprocess
from sklearn.mixture import GaussianMixture

os.makedirs('./output/geyser', exist_ok=True)
sns.set(color_codes=True)
sns.set(font_scale=1.5)

def read_file(filename):    
    infile = open(filename, "r")
    for ligne in infile:
        if ligne.find("eruptions waiting") != -1:
            break

    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [float(x) for x in ligne.split()]
        data.append(eruption)
        data.append(waiting)
    infile.close()

    data = np.asarray(data).reshape(-1, 2)
    return data

import numpy as np

data = read_file('../data/geyser.txt')
plt.figure(figsize=(10,8))
sns.scatterplot(data.T[0], data.T[1], s=200)
plt.title('Eruptions at old faithful')
plt.xlabel('Duration of eruption (minutes)')
plt.ylabel('Waiting time (minutes)')
plt.tight_layout()
plt.savefig('output/geyser/geyser.png')

def draw(i, tag, gmm):
    sns.set(font_scale=3.0)
    gmm.fit(data)
    sample_points = gmm.sample(500)[0]
    fig, ax = plt.subplots(figsize=(16,12))

    x = data.T[0]
    y = data.T[1]

    means = gmm.means_
    g = sns.kdeplot(sample_points.T[0], sample_points.T[1], shade=True)

    ax.scatter(x, y, marker='x', s=200, cmap='jet')
    colors = ['darkblue', 'darkred']
    for j in range(means.shape[0]):
        plt.text(means[j, 0], means[j, 1], f'Centroid {j}', size=30, color=colors[j])
    
    plt.xlim([x.min(), x.max()])
    plt.ylim([y.min(), y.max()])

    plt.title('Eruptions at old faithful')
    plt.xlabel('Duration of eruption (minutes)')
    plt.ylabel('Waiting time (minutes)')
    plt.tight_layout()

    plt.savefig(f'output/geyser/{tag}/{i:04d}.png')
    
def generate_videos(n_components):
    np.random.seed(0)

    tag = f"{n_components}_gmm"
    os.makedirs(f'output/geyser/{tag}', exist_ok=True)
    means = np.array(
        [
            [2.5, 70.0],
            [4.5, 90.0]
        ]
    )

    gmm = GaussianMixture(
        n_components=n_components, 
        max_iter=1, 
        init_params='random', 
        warm_start=True,
        covariance_type='tied',
        means_init=means
    )

    for i in tqdm(range(9)):
        means = draw(i, tag, gmm)

    command = (
        f"ffmpeg -y -r 2 -f image2 -s 1920x1080 -i ./output/geyser/{tag}/%04d.png "
        f"-vcodec libx264 -crf 10 -pix_fmt yuv420p "
        f"./output/geyser/{tag}.mp4"
    )
    command = command.split(' ')
    subprocess.call(command)

n_components = [2]

for n in n_components:
    generate_videos(n)