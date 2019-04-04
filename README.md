# Generalized Variational Inference for DGPs and BNNs

The README.md files inside the two folders provide more detail. Note that you will need two different virtual python environments for each. If you have never set up and installed different package versions into different python environments before, comprehensive instructions for doing so within anaconda are available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Note that the provided code it the exact same code that was used to produce the results in Knoblauch, Jewson & Damoulas (2019), and we kindly ask you to cite this paper if you are using our code base. If you are using the DGP component of the code base, it is appropriate to also cite Salimbeni & Deisenroth (2017).

*Important*: For both subfolders BNN and DSDGP, you will have to unzip 'data.zip' before you can execute any of the experiments.

The Deep Gaussian Process code works with Python 3.6, tensorflow 1.8.0 and gpflow 1.1.1, but may fail for both older and more recent versions. 

The Bayesian Neural Network code works with Python 2.7.14 and autograd, but may fail for more recent versions

This code was produced as part of the following paper:

@article{GVI,
  title={Generalized Variational Inference},
  author={Knoblauch, Jeremias and Jewson, Jack and Damoulas, Theodoros},
  journal={arXiv preprint arXiv:1904.02063},
  year={2019}
}

