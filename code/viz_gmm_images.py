import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
import numpy as np
import glob, os
import seaborn as sns


from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

import IPython
import subprocess

plt.style.use('seaborn-poster')
sns.set(color_codes=True)

def get_image(file_path):
    im = Image.open(file_path)
    colors = im.histogram()
    return np.asarray(im)

images = []
labels = []
paths = []
for path in glob.glob('../data/images/*/*.jpeg', recursive=True):
    data = get_image(path).reshape(-1, 3).copy()
    #data = data - data.mean(axis=0)
    #data = data / data.var(axis=0)
    image = np.median(data, axis=0)
    labels.append('forests' in path)
    images.append(image)
    paths.append(path)
images = np.array(images)

def getImage(path,predicted_label,zoom=0.2):
    actual_label = 'forests' in path
    im = plt.imread(path)    
    return OffsetImage(im,zoom=zoom)

def frame_image(img, frame_width):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    if img.ndim == 3: # rgb or rgba array
        framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]))
    elif img.ndim == 2: # grayscale image
        framed_img = np.zeros((b+ny+b, b+nx+b))
    framed_img[b:-b, b:-b] = img
    return framed_img

def draw(i, i1, i2, gmm, tag):
    sns.set(font_scale=3.0)
    gmm.fit(images)
    sample_images = gmm.sample(1000)[0]

    color_labels = ['red', 'green', 'blue']
    
    fig, ax = plt.subplots(figsize=(16,12))
    g = sns.kdeplot(sample_images.T[i1], sample_images.T[i2], shade=True)
    x = images.T[i1]
    y = images.T[i2]
    ax.scatter(x, y, c=labels, marker='x')
    predicted_labels = gmm.predict(images)

    artists = []
    for x0, y0, path, predicted_label in zip(x, y, paths, predicted_labels):
        ab = AnnotationBbox(getImage(path, predicted_label), (x0, y0), frameon=False)
        artists.append(ax.add_artist(ab))
    
    slackx = .1*x.mean()
    slacky = .1 * y.mean()
    plt.xlim([x.min() - slackx, x.max() + slackx])
    plt.ylim([y.min() - slacky, y.max() + slacky])

    if gmm.n_components == 2:
        accuracies = [accuracy_score(labels, predicted_labels), accuracy_score(labels, 1 - predicted_labels)]
        accuracy = max(accuracies)
        plt.text(70, 120, f'Accuracy: {accuracy*100:.2f}%', size=30, color='darkblue')

    plt.xlabel(f'Median {color_labels[i1]} value in RGB')
    plt.ylabel(f'Median {color_labels[i2]} value in RGB')
    
    plt.tight_layout()
    plt.savefig(f'output/images/{tag}/{i:04d}.png')
    
def generate_videos(n_components, covariance_type, init_method):
    np.random.seed(0)

    gmm = GaussianMixture(
        n_components=n_components, 
        max_iter=1, 
        init_params=init_method, 
        warm_start=True,
        covariance_type=covariance_type
    )
    tag = f"{covariance_type}_{n_components}_{init_method}"
    os.makedirs(f'output/images/{tag}/', exist_ok=True)

    for i in tqdm(range(25)):
        draw(i, 1, 2, gmm, tag)

    command = (
        f"ffmpeg -y -r 2 -f image2 -s 1920x1080 -i ./output/images/{tag}/%04d.png "
        f"-vcodec libx264 -crf 10 -pix_fmt yuv420p "
        f"./output/images/{tag}.mp4"
    )
    command = command.split(' ')
    subprocess.call(command)

n_components = [2]
covariance_type = ['spherical']
init_methods = ['random']

for n in n_components:
    for cov in covariance_type:
        for init_method in init_methods:
            generate_videos(n, cov, init_method)