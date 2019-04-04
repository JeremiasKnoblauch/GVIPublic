# Generalized Variational Inference for Bayesian Neural Nets

This code is a merged and strongly modified version of the code repos build by Li & Turner '16 and Hernandez-Lobato et al. '16. 

### Compatibility

Note that the current implementation requires Spyder 2.7.14 and autograd to work.


### Running the paper experiments

The code can be run via 'run\_BNN.py' either via an IDE (by adapting lines 197-220 appropriately) or via the terminal. The script implements some superficial error-checking to ensure that input is reasonable and will abort or give warnings if it is not. If run from the command line, the script takes 5 arguments.

If running from the terminal, the correct order of the arguments of 'run\_BNN.py' are

1. dataset, which can be one of ['boston', 'concrete', 'energy', 'kin8mn', 'naval', 'power', 'wine', 'yacht'].
2. Dtype, which gives the inference method used and can be one of ['KL', 'AR', 'A', 'A-approx', 'AR-approx', 'AB-approx']. We explain these below.
3. alpha, which is a float whose meaning depends on Dtype. If Dtype = 'KL', alpha will be a void/unused argument. To ensure that it is obvious when this is the case, the script automatically sets alpha = -99 if Dtype = 'KL' unless you do it yourself. We note that you will still have to specify a value for alpha, and we recommend that you choose alpha = -99 if Dtype = 'KL'.
4. losstype, which can be one of ['b', 'g', 'standard']. Here, 'b' and 'g' stand for the beta- and gamma-divergence induced losses respectively. 'standard' is the usual negative log likelihood loss (i.e. standard Bayesian inference)
5. loss\_hyperparam, which is beta for losstype = 'b', gamma for losstype = 'g'. If Dtype = 'AB-approx', this parameter takes the role of the second (the beta) argument of the scaled alpha-beta-divergence. We explain this further below.

##### More detail on Dtype

'KL' corresponds to standard Variational Inference and the -approx indicates an F-VI method. 'AR-approx' is the alpha-renyi based F-VI method of Li & Turner '16, 'A-approx' the alpha-divergence based F-VI method of Hernandez-Lobato '16, 'AB-approx' the (scaled) alpha-beta-divergence F-VI method of Regli & Silva '18. For 'AB-approx', alpha specifies the first (the alpha (A)) argument of the divergence, and loss\_hyperparam will take the role of the second (the beta (B)) argument of the divergence. 
'AR' corresponds to GVI with the Renyi-alpha uncertainty quantifier. 'A' corresponds to GVI with the alpha-divergence uncertainty quantifier. Note that is not recommended, as it uses the (bounded) alpha-divergence as uncertainty quantifier and thus overconcentrates massively. (See also the additional pictures in the appendix.)

##### Running the script

Summarizing the above, the script is called via

```
python run_BNN.py (dataset) (Dtype) (alpha) (losstype) (loss_hyperparam)
```
