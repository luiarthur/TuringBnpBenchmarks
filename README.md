# TuringBnpBenchmarks
Benchmarks of Bayesian Nonparametric models in Turing and other PPLs.

This work is funded by [GSoC 2020][1].

My mentors for this project are [Hong Ge][3], [Martin Trapp][4], and 
[Cameron Pfiffer][5].

## Abstract
Probabilistic models, which more naturally quantify uncertainty when compared
to their deterministic counterparts, are often difficult and tedious to
implement. Probabilistic programming languages (PPLs) have greatly increased
productivity of probabilistic modelers, allowing practitioners to focus on
modeling, as opposed to the implementing algorithms for probabilistic (e.g.
Bayesian) inference. Turing is a PPL developed entirely in Julia and is both
expressive and fast due partly to Julia’s just-in-time (JIT) compiler being
implemented in LLVM. Consequently, Turing has a more manageable code base and
has the potential to be more extensible when compared to more established PPLs
like STAN. One thing that may lead to the adoption of Turing is more benchmarks
and feature comparisons of Turing to other mainstream PPLs. The aim of this
project is to provide a more systematic approach to comparing execution times
and features among several PPLs, including STAN, Pyro, nimble, and Tensorflow
probability for a variety of Bayesian nonparametric (BNP) models, which are a
class of models that provide a much modeling flexibility and often allow model
complexity to increase with data size.

To address the need for a more systematic approach for comparing the
performance of Turing and various PPLs (STAN, Pyro, nimble, TensorFlow
probability) under common Bayesian nonparametric (BNP) models,  which are a
class of models that provide a great deal of modeling flexibility and allow the
number of model parameters, and thus model complexity, to increase with the
size of the data. The following models will be implemented (if possible) and
timed (both compile times and execution times) in the various PPLs (links to
minimum working examples will be provided):

- Sampling (and variational) algorithms for Dirichlet process (DP) Gaussian /
  non-Gaussian mixtures for different sample sizes
    - E.g. Sampling via Chinese restaurant process (CRP) representations
      (including collapsed Gibbs, sequential Monte Carlo, particle Gibbs),
      HMC/NUTS for stick-breaking (SB) constructions, variational inference for
      stick-breaking construction.
    - **Note**: DPs are a popular choice of BNP models typically used when density
      estimation is of interest. They are also a popular prior for infinite
      mixture models, where the number of clusters are not known in advance.
- Sampling (and variational) algorithms for Pitman-Yor process (PYP) Gaussian /
  non-Gaussian mixtures for different sample sizes
    - E.g. Sampling via generalized CRP representations (including collapsed
      Gibbs, sequential Monte Carlo, particle Gibbs), HMC/NUTS for
      stick-breaking (SB) constructions, variational inference for
      stick-breaking construction.
    - **Note**: PYPs are generalizations of DPs. That is, DPs are a special
      case of PYPs. PYPs exhibit a power-law behavior, which enables them to
      better model heavy-tailed distributions.
- PYP / DP hierarchical models. Specific model to be determined.

In addition, the effective sample size and inference speed of a standardised
setup, e.g. HMC in truncated stick-breaking DP mixture models, for the
respective PPLs will be measured.

## What this repo contains
This repository includes (or will include) tables and other visualizations
that compare the (compile and execution) speed and features of various PPLs
(Turing, STAN, Pyro, Nimble, TFP) with a repository containing the minimum
working examples (MWEs) for each implementation. Blog posts describing the
benchmarks will also be included.

## Software / Hardware
All experiments for this project were done in an [c5.xlarge][2] AWS Spot
Instance. As of this writing, here are the specs for this instance:

- vCPU: 4 Intel(R) Xeon(R) Platinum 8124M CPU @ 3.00GHz
- RAM: 8 GB
- Storage: EBS only
- Network Bandwidth: Up to 10 Gbps
- EBS Bandwidth: Up to 4750 Mbps

The following software was used:
- Julia-v1.4.1. See `Project.toml` and `Manifest.tomal` for more info.

[1]: https://summerofcode.withgoogle.com/projects/#5861616765108224
[2]: https://aws.amazon.com/ec2/instance-types/c5/
[3]: http://mlg.eng.cam.ac.uk/hong/ 
[4]: https://martintdotblog.wordpress.com/
[5]: http://cameron.pfiffer.org/
