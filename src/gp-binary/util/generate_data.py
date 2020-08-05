import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

np.random.seed(1)

def plot_data(x, y):
    colors = {1: 'red', 0: 'blue'}
    clabel = [colors[yi] for yi in y]
    plt.scatter(x[:, 0], x[:, 1], c=clabel)

for nobs in (50, 100, 200, 300):
    x, y = make_moons(n_samples=nobs, shuffle=True, noise=0.1, random_state=1)
    output = dict(x1=x[:, 0].tolist(),
                  x2=x[:, 1].tolist(),
                  y=y.tolist())
    save_path = '../data/gp-binary-data-N{}.json'.format(nobs)
    with open(save_path, 'w') as outfile:
        json.dump(output, outfile)

    print('Created data with {} observations.'.format(nobs))
