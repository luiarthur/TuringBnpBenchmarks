import numpy as np
import json

# TODO: Generate data
np.random.seed(1)

# True function.
def f(x):
    return np.sin(3 * x) * np.sin(x) * (-1)**(x > 0)

def make_data(nobs, ngrid=1000):
    """
    nobs (int): number of observations.
    ngrid (int): number of grid points in fine grid for plotting true function.
    """
    # Predictors (between -3 and 3).
    x = np.random.rand(nobs) * 6 - 3

    # Response.
    y = f(x)

    # Finer grid for plotting true function.
    x_true = np.linspace(-3.5, 3.5, ngrid)
    f_true = f(x_true)

    # Create output
    output = dict(x=x.round(3).tolist(),
                  f=y.round(3).tolist(),
                  x_true=x_true.round(3).tolist(),
                  f_true=f_true.round(3).tolist())

    return output

# Save data
for nobs in (30, 100, 300):
    output = make_data(nobs)
    save_path = '../data/gp-data-N{}.json'.format(nobs)
    with open(save_path, 'w') as outfile:
        json.dump(output, outfile)

    print('Created data with {} observations.'.format(nobs))

    # Read data by:
    # with open(save_path, 'r') as f:
    #     data = json.loads(f.read())
