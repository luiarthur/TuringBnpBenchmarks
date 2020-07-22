import numpy as np
import json

# TODO: Generate data
np.random.seed(1)

# True function.
def f(x):
    return np.sin(3 * x) * np.sin(x) * (-1)**(x > 0)

# Number of observations.
N = 30

# Predictors (between -3 and 3).
x = np.random.rand(N) * 6 - 3

# Response.
y = f(x)

# Finer grid for plotting true function.
x_true = np.linspace(-3.5, 3.5, 1000)
f_true = f(x_true)

# Create output
output = dict(x=x.round(3).tolist(),
              f=y.round(3).tolist(),
              x_true=x_true.round(3).tolist(),
              f_true=f_true.round(3).tolist())

# Save data
save_path = '../data/gp-data-N{}.json'.format(N) 
with open(save_path, 'w') as f:
    json.dump(output, f)

# Read data by:
# with open(save_path, 'r') as f:
#     data = json.loads(f.read())

