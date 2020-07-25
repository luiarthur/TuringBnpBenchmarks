import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Squared exponential covariance function (for stationary GP).
def sqexp_cov_fn(d, rho, alpha):
    return alpha ** 2 * np.exp(-0.5 * (d / rho) ** 2)
    
# Function to create gp prediction function.
def gp_predict_maker(y, x, x_new, cov_fn):
    N = x.shape[0]
    N_new = x_new.shape[0]
    M = N + N_new
    xx = np.concatenate((x_new, x)).reshape(M, 1)
    D = distance_matrix(xx, xx)
    
    # Function which takes parameters of covariance function
    # and predicts at new locations.
    def gp_predict(rho, alpha, eps):
        K = cov_fn(D, rho, alpha) + np.eye(M) * eps
        K_new_old = K[:N_new, N_new:]
        K_old_inv = np.linalg.inv(K[N_new:, N_new:])
        C = K_new_old.dot(K_old_inv)
        mu = C.dot(y)
        S = K[:N_new, :N_new] - C.dot(K_new_old.T)
        return np.random.multivariate_normal(mu, S)
    
    return gp_predict

# Function for plotting parameter posterior.
def plot_post(samples, key, bins=None, suffix=""):
    plt.hist(samples[key], density=True, bins=bins)
    plt.xlabel(key)
    plt.ylabel('density')
    if suffix != "":
        suffix = "({})".format(suffix)
    
    plt.title("{} {}".format(key, suffix));
    
# Function for making all plots.
def make_plots(samples, x, y, x_true, f_true, cov_fn=sqexp_cov_fn,
               n_new=100, figsize=(12,4), figsize_f=(12, 4), suffix="",
               x_min=-3.5, x_max=3.5, ci=95, eps=1e-3):
    # Create new locations for prediction.
    # But include the data for illustrative purposes.
    x_new = np.linspace(x_min, x_max, n_new)
    x_new = np.sort(np.concatenate((x_new, x)))

    # Create gp predict function.
    gp_predict = gp_predict_maker(y, x, x_new, cov_fn)

    # Number of posterior samples.
    nsamples = len(samples['alpha'])

    # Make predictions at new locations.
    preds = np.stack([gp_predict(alpha=samples['alpha'][b],
                                 rho=samples['rho'][b],
                                 eps=eps)
                      for b in range(nsamples)])
      
    # Plot parameters posterior.
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plot_post(samples, 'alpha', bins=30, suffix=suffix)
    plt.subplot(1, 2, 2)
    plot_post(samples, 'rho', bins=30, suffix=suffix)
    
    # Summarize function posterior.
    ci_lower = (100 - ci) / 2
    ci_upper = (100 + ci) / 2
    preds_mean = preds.mean(0)
    preds_lower = np.percentile(preds, ci_lower, axis=0)
    preds_upper = np.percentile(preds, ci_upper, axis=0)
    
    # Make suffix
    if suffix != "":
        suffix = "({})".format(suffix)

    # Plot function posterior.
    plt.figure(figsize=figsize_f)
    plt.scatter(x, y, c='black', zorder=3, label='data')
    plt.fill_between(x_new, preds_upper, preds_lower, alpha=.3, label='95% CI')
    plt.plot(x_new, preds.mean(0), lw=2, label="mean fn.")
    plt.plot(x_true, f_true, label="truth", lw=2, c='red', ls=':')
    plt.title("GP Posterior Predictive with 95% CI {}".format(suffix))
    plt.legend(); 

