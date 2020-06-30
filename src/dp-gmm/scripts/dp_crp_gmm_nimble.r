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

renv::activate("../../..")  # activate the project env.

library(nimble)
library(rjson)

# NOTE: From NIMBLE docs: https://r-nimble.org/html_manual/cha-bnp.html
#
# Nimble doesn't allow dynamic allocation of new clusters. So, K is an upperbound
# for the number of clusters for computational and memory convenience. K should
# be much less than N.
model.code = nimbleCode({
    # Class Labels
    z[1:N] ~ dCRP(alpha, size=N)
    
    # DP concentration parameter
    alpha ~ dgamma(1, 100)
    
    for(k in 1:K) {
        mu[k] ~ dnorm(0, 3)
        sig2[k] ~ dinvgamma(2, 0.05)
    }
    
    for(i in 1:N) {
        y[i] ~ dnorm(mu[z[i]], var=sig2[z[i]])  
    }
})

# Give the input file name to the function.
path_to_simdata = "../../data/sim-data/gmm-data-n200.json"
simdata = fromJSON(file=path_to_simdata)
hist(simdata$y, breaks=30)

set.seed(2)

data = list(y=simdata$y)
constants = list(N=length(data$y), K=10)
inits = list(mu=rnorm(constants$K, 0, 3),
             sig2=rep(0.01, constants$K),
             z=rep(1, constants$N),
             alpha=0.1)

# burn = 1000
# nsamples = 1000
# samples = nimbleMCMC(code=model.code, 
#                      constants=constants, 
#                      data=data, 
#                      inits=inits,
#                      monitors=c('mu', 'sig2', 'alpha', 'z'), setSeed=1,
#                      nburnin=burn, niter=burn + nsamples, progressBar=TRUE)

system.time({
    dp_crp_gmm <- nimbleModel(model.code, constants, data, inits)
    compiled_dp_crp_gmm <- compileNimble(dp_crp_gmm)
    mcmc <- buildMCMC(compiled_dp_crp_gmm, monitors=c('alpha', 'sig2', 'mu', 'z'))
    compiled_dp_crp_gmm <- compileNimble(mcmc)
})

system.time({
    samples <- runMCMC(compiled_dp_crp_gmm, nburnin=1000, niter=2000, setSeed=1)
})

colnames(samples)
dim(samples)

N = constants$N
mu = samples[, 2:11]
sig2 = samples[, 12:21]
alpha = samples[, 'alpha']
z = samples[, 22:(22+N-1)]

num_clus = apply(z, 1, function(row) length(unique(row)))
                 
options(repr.plot.width=15, repr.plot.height=11)
par(mfrow=c(2,2))
plot(table(num_clus)/NROW(num_clus), xlab="Number of Clusters", ylab="Posterior Probability")

# NOTE: Some of the components do not get updated under the CRP
# representation because they never get used.                 
boxplot(mu, main="Boxplot of mu")
abline(h=simdata$mu, lty=2)

boxplot(sqrt(sig2), main="Boxplot of sigma")
abline(h=simdata$sig, lty=2)

hist(alpha, prob=TRUE)


