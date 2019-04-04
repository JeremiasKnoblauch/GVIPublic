# Generalized Variational Inference for Doubly-Stochastic Deep Gaussian Processes

This code is built on the core code of Salimbeni & Deisenroth '17. Specifically, all files in the folder 'doubly_stochastic_dgp' as well as 'datasets.py' are modifications of their code base, which is accessible [here](https://github.com/ICL-SML/Doubly-Stochastic-DGP). At the top of each file, I explain which lines have been altered and to what purpose.

### Compatibility: 

The code works with Python 3.6, tensorflow 1.8.0 and gpflow 1.1.1, but may fail for more recent versions. 

### Portability:

The code should be very portable to gpflow more generally. Specifically, 'robustified_likelihood.py' will work with any type of GP model inside gpflow.

### Running the paper experiments

The code can be run via 'run_DGP.py' either via an IDE (by adapting lines 225-249 appropriately) or via the terminal. The script implements some superficial error-checking to ensure that input is reasonable and will abort or give warnings if it is not. If run from the command line, the script takes 5 arguments.

If running from the terminal, the correct order of the arguments of 'run_DGP.py' are

1. dataset, which can be one of ['boston', 'concrete', 'energy', 'kin8mn', 'naval', 'power', 'protein', 'wine', 'yacht'].
2. alphas, which is a list of *floats* s.t. the l-th entry of that list provides the value of alpha in the alpha-renyi divergence uncertainty quantifier for the l-th layer. If alpha = 1.0, this will result in the Kullback-Leibler (i.e. standard) uncertainty quantification. On the terminal, this should be specified as e.g. 1.0,1.0,0.5 to give an alpha-renyi with alpha = 0.5 in the third layer and the standard KL uncertainty quantifiers in the first two layers.
3. div_weights, which is a list of *floats* s.t. the l-th entry of that list provides the weight by which the l-th uncertainty quantifier is multiplied. (NOTE: Unlike in the paper, this is NOT the w from 1/w! This is what you you multiply the uncertainty quantifier with directly. I.e., if I specify w = 3, I will use the uncertainty quantifier 3 * D.)
4. losstype, which can be one of ['b', 'g', 'standard']. Here, 'b' and 'g' stand for the beta- and gamma-divergence induced losses respectively. 'standard' is the usual negative log likelihood loss (i.e. standard Bayesian inference).
5. loss_hyperparam, which is beta for losstype = 'b', gamma for losstype = 'g'. Set to -99 if losstype = 'standard'. If losstype is neither of the two, then if you don't do it yourself, this parameter will be set to -99 by the script and you will receive a warning about it.


##### Running the script

Summarizing the above, the script is called via

```
python run_DGP.py (dataset) (alphas) (div_weights) (losstype) (loss_hyperparam)
```

