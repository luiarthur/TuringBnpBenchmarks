import numpy as np
import matplotlib.pyplot as plt
import json

# Read GP data.
path_to_data = '../../../../src/gp/data/gp-data-N30.json'
simdata = json.load(open(path_to_data))

# Plot data and true function.
nobs = len(simdata['f'])

for add_noise in (False, True):
    if add_noise:
        noise = np.random.randn(nobs) * 0.1
    else:
        noise = np.zeros([])

    plt.scatter(simdata['x'], simdata['f'] + noise, label='data')
    plt.plot(simdata['x_true'], simdata['f_true'], ls=':', c='grey', label='true f(x)')
    plt.xlabel('x')
    plt.ylabel('y = f(x)')
    plt.legend()
    plt.savefig('gp-data{}.png'.format('-noisy' if add_noise else ''), bbox_inches='tight')
    plt.close()

# Linear Data
np.random.seed(1)
nobs = 100
x = np.sort(np.random.randn(nobs))
f = lambda x: x * 3 + 2
y = f(x) + np.random.randn(nobs) * 1
plt.scatter(x, y, label='data')
plt.plot(x, f(x), label='truth', color='grey' , ls=":")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('linear-data.png')
plt.close()
