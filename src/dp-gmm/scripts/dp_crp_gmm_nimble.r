cat("Last updated:", date(), "(PT)")

# The R-package `renv` was used to create an R virtual environment.
# To install `renv`, do `install.packages('renv')`.
# To initialize an R environment, `renv::init()`.
# This will create a directory called `renv` in the current dir.

# See also:
# renv::init()
# renv::activate()
# renv::settings$snapshot.type('all')
# renv::status()
# renv::snapshot()
# renv::restore()

# Activate the virtual env.
renv::activate("../../..")

# Import libraries
library(nimble)
library(rjson)

# Define DP mixture of Gaussians model

# NOTE: From NIMBLE docs: https://r-nimble.org/html_manual/cha-bnp.html
#
# Nimble doesn't allow dynamic allocation of new clusters. So, K is an upperbound
# for the number of clusters for computational and memory convenience. K should
# be much less than N.
model.code = nimbleCode({
    # CRP priors of class labels.
    z[1:N] ~ dCRP(alpha, size=N)
    
    # DP concentration parameter.
    alpha ~ dgamma(1, 100)
    
    # Base distribution.
    # Note that dynamic allocation of clusters is not allowed.
    # So, an upperbound (K) is specifeid.
    for(k in 1:K) {
        mu[k] ~ dnorm(0, 3)
        sig2[k] ~ dinvgamma(2, 0.05)
    }
    
    # Sampling distribution.
    for(i in 1:N) {
        y[i] ~ dnorm(mu[z[i]], var=sig2[z[i]])  
    }
})

# Read and visualize data.
path_to_simdata = "../data/gmm-data-n200.json"
simdata = fromJSON(file=path_to_simdata)
hist(simdata$y, breaks=30)

# Set seed for reproducibility.
set.seed(2)

# Define data.
data = list(y=simdata$y)

# Define model constants.
constants = list(N=length(data$y), K=10)

# Initialize model (optional).
inits = list(mu=rnorm(constants$K, 0, 3),
             sig2=rep(0.01, constants$K),
             z=rep(1, constants$N),
             alpha=0.1)

# One liner version.

# burn = 1000
# nsamples = 1000
# samples = nimbleMCMC(code=model.code, 
#                      constants=constants, 
#                      data=data, 
#                      inits=inits,
#                      monitors=c('mu', 'sig2', 'alpha', 'z'), setSeed=1,
#                      nburnin=burn, niter=burn + nsamples, progressBar=TRUE)

# Compile model.

system.time({
    dp_crp_gmm <- nimbleModel(model.code, constants, data, inits)
    compiled_dp_crp_gmm <- compileNimble(dp_crp_gmm)
    mcmc <- buildMCMC(compiled_dp_crp_gmm, monitors=c('alpha', 'sig2', 'mu', 'z'))
    compiled_dp_crp_gmm <- compileNimble(mcmc)
})

# Run MCMC and get samples from joint posterior.

system.time({
    samples <- runMCMC(compiled_dp_crp_gmm, nburnin=1000, niter=2000, setSeed=1)
})

# See names of parameters and dimensions.
colnames(samples)
dim(samples)

# Number of observations.
N = constants$N

# Posterior of mu
mu = samples[, 2:11]

# Posterior of sigma^2
sig2 = samples[, 12:21]

# Posterior of concentration parameter
alpha = samples[, 'alpha']

# Posterior of class labels
z = samples[, 22:(22+N-1)]

# Posterior of number of clusters
num_clus = apply(z, 1, function(row) length(unique(row)))
                 
                 

# Plot posteriors
options(repr.plot.width=15, repr.plot.height=11)
par(mfrow=c(2, 2))

# Plot posterior probability of number of clusters
plot(table(num_clus)/NROW(num_clus), xlab="Number of Clusters", ylab="Posterior Probability")

# Plot marginal posterior distribution of each mu component.
# NOTE: Some of the components do not get updated under the CRP
# representation because they never get used.
boxplot(mu, main="Boxplot of mu")
abline(h=simdata$mu, lty=2)

# Plot marginal posterior distribution of each sigma component.
boxplot(sqrt(sig2), main="Boxplot of sigma", ylim=c(0, 0.4))
abline(h=simdata$sig, lty=2)

# Plot posterior distribution of each alpha.
hist(alpha, prob=TRUE)
par(mfrow=c(1, 1))

# Sanity check. Plot proportion of mixture components for one sample of z.
# They should match the simulation truth.
options(repr.plot.width=6, repr.plot.height=6)
plot(table(z[1000,])/constants$N, xlab="mixture component", ylab="count", main="proportion for one sample from posterior")
abline(h=simdata$w, lty=2)  # simulation truth.


