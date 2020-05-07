# TuringBnpBenchmarks
Benchmarks of Bayesian Nonparametric models in Turing and other PPLs.

This work is funded by [GSoC 2020][1].

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

## What this repo contains
This repository includes (or will include) tables and other visualizations
that compare the (compile and execution) speed and features of various PPLs
(Turing, STAN, Pyro, Nimble, TFP) with a repository containing the minimum
working examples (MWEs) for each implementation. Blog posts describing the
benchmarks will also be included.

[1]: https://summerofcode.withgoogle.com/projects/#5861616765108224
