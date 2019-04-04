from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad

import sys

import math




"""TOP LEVEL FUNCTION. Start reading here"""
def fit_q(X, y, hidden_layer_size, batch_size, epochs, K, location, scale, 
          alpha = 1.0, 
          learning_rate = 1e-2, v_prior = 1.0, Dtype="AR", losstype = None, 
          beta_L = None, gamma_L = None,
          beta_D = -0.3, m_prior = 0.0, 
          X_validation=None, y_validation=None
          ):
    
    """This function wraps everything. It can be broken up into three parts: 
        (1) Setting up the BNN model;
        (2) Creating the functional counterparts of the BNN;
        (3) Running stochastic optimization for the BNN.
    """


    """STEP (1) + (2): Set up the BNN model and create the relevant functional
                        objects via 'make_functions' 
    """
    hidden_layer_size = [ hidden_layer_size ]
    hidden_layer_size = np.array([ X.shape[ 1 ] ] + hidden_layer_size + [ 1 ])
    shapes = zip(hidden_layer_size[ : -1 ], hidden_layer_size[ 1 : ])
    w, energy, get_error_and_ll, get_predictive_distribution = make_functions(
            X.shape[ 1 ], shapes, alpha, Dtype, 
            losstype, beta_L, gamma_L, beta_D)
    
    #Obtain the gradient of our objective function (will depend on Dtype etc.)
    energy_grad = grad(energy)

    print("    Epoch      |    Error  |   Log-likelihood  ")

    def print_perf(epoch, w):
        error, ll = get_error_and_ll(w, v_prior, X, y, K, 0.0, 1.0)
        print("{0:15}|{1:15}|{2:15}".format(epoch, error, ll))
        sys.stdout.flush()


    """STEP (3) Iterate #epoch=500 times with baches of size 32 and learning
                rate 0.001. Use ADAM with default settings for the optimization
    """
    # Train with sgd
    batch_idxs = make_batches(X.shape[0], batch_size)

    m1 = 0
    m2 = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    t = 0

    #for each epoch, loop over all the batches in your permutation
    for epoch in range(epochs):
        permutation = np.random.choice(range(X.shape[ 0 ]), X.shape[ 0 ], replace = False)
        print_perf(epoch, w)
        
        #For each batch, do one step
        for idxs in batch_idxs:
            
            #Learn parameters with ADAM.
            t += 1
            grad_w = energy_grad(w, X[ permutation[ idxs ] ], y[ permutation[ idxs ] ], 
                                 v_prior, m_prior, K, X.shape[ 0 ], alpha)
            m1 = beta1 * m1 + (1 - beta1) * grad_w
            m2 = beta2 * m2 + (1 - beta2) * grad_w**2
            m1_hat = m1 / (1 - beta1**t)
            m2_hat = m2 / (1 - beta2**t)
            w -= learning_rate * m1_hat / (np.sqrt(m2_hat) + epsilon)
            

    return w, v_prior, get_error_and_ll, get_predictive_distribution



"""AUXILIARY OBJECT. Purpose is to provide wrapper for all params"""
class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += np.prod(shape)
        self.idxs_and_shapes[ name ] = (slice(start, self.num_weights), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[ idxs ], shape)

    def get_indexes(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return idxs
 
"""AUXILIARY FUNCTION. Purpose is to slice the data into batches (for fit_q)"""
def make_batches(N_data, batch_size):
    return [ slice(i, min(i + batch_size, N_data)) for i in range(0, N_data, batch_size) ]



"""WRAPPER FUNCTION. 
    Responsible for the creation of all functions that make
    inference possible in the BNN. Creates functions according
    to specifications passed down from 'fit_q'.
                     
    Legend: Since some functions are only used for certain F-VI methods,
        we label each function according to which inference method
        relies on it.
        'UNIVERSAL': all inference methods rely on a function.
        'A-APPROX': Only Hernandez-Lobato et al. '16 (black box alpha-div)
        'AR/AB-APPROX': Only Li & Turner '16; Regli & Silva '18 
                        (Renyi-alpha-div; scaled alpha-beta-div)
        'GVI': Only needed for Dtype = AR (i.e., alpha-renyi uncertainty quantification)
"""
def make_functions(d, shapes, alpha, Dtype="AR", 
                   losstype = None, beta_L = None, gamma_L = None, beta_D = -0.3):
    N = sum((m + 1) * n for m, n in shapes)
    parser = WeightsParser()
    parser.add_shape('mean', (N, 1))
    parser.add_shape('log_variance', (N, 1))
    parser.add_shape('log_v_noise', (1, 1))

    w = 0.1 * np.random.randn(parser.num_weights)
    w[ parser.get_indexes(w, 'log_variance') ] = w[ parser.get_indexes(w, 'log_variance') ] - 10.0
    w[ parser.get_indexes(w, 'log_v_noise') ] = np.log(1.0)

    
    """UNIVERSAL. 
        produces (stochastic) predictions from BNN by sampling from
        the variational posterior q, given a set of inputs X.
    """
    def predict(samples_q, X):

        # First layer

        K = samples_q.shape[ 0 ]
        (m, n) = shapes[ 0 ]
        W = samples_q[ : , : m * n ].reshape(n * K, m).T
        b = samples_q[ : , m * n : m * n + n ].reshape(1, n * K)
        a = np.dot(X, W) + b
        h = np.maximum(a, 0)

        # Second layer

        samples_q = samples_q[ : , m * n + n : ]
        (m, n) = shapes[ 1 ]
        b = samples_q[ : , m * n : m * n + n ].T
        a = np.sum((samples_q[ : , : m * n ].reshape(1, -1) * h).reshape((K * X.shape[ 0 ], m)), 1).reshape((X.shape[ 0 ], K)) + b

        return a

    """UNIVERSAL. 
        produces the relevant (pseudo) likelihooods depending on the specified
        loss. I.e., for loss l(x_i, \theta), this produces 
        exp(-sum_{i=1}^n l(x_i, \theta)).
    """
    def log_likelihood_factor(samples_q, v_noise, X, y):
        """NOTE: Might have to remove the prior from this!"""
        #lkl = -0.5 * np.log(2 * math.pi * v_noise) - 0.5 * (np.tile(y, (1, samples_q.shape[ 0 ])) - outputs)**2 / v_noise
        outputs = predict(samples_q, X)
        log_prior_noise_factor = -0.5 * np.log(2 * math.pi * v_noise)
        #print("y shape", y.shape)
        log_obs_factor = - 0.5 * (np.tile(y, (1, samples_q.shape[ 0 ])) - outputs)**2 / v_noise
        if (losstype is None or losstype == "standard"):
            """standard normal likelihood"""
            log_lkl = log_prior_noise_factor + log_obs_factor
        elif (losstype == "beta" or losstype == "b") and beta_L is not None:
            """beta-divergence transformed normal likelihood, see e.g.
            Appendix of Knoblauch, Jewson and Damoulas (2018)"""
            sig_beta = (v_noise * 2 * np.pi) ** (- beta_L * 0.5)
            integral_term = (1.0 + beta_L) ** (-1.5) * sig_beta
            log_lkl =( log_prior_noise_factor - integral_term + 
                    (1.0 / beta_L) * sig_beta * np.exp(beta_L * log_obs_factor))
        elif (losstype == "gamma" or losstype == "g") and gamma_L is not None:
            """gamma-divergence transformed normal likelihood, see 
            Fujisawa & Eguchi (2018)"""
            sig_gamma = (v_noise * 2 * np.pi) ** (- gamma_L * 0.5)
            integral_term = (1.0 + gamma_L) ** (-1.5) * sig_gamma
            log_lkl =( log_prior_noise_factor + 
                    (1.0 / gamma_L) * sig_gamma * np.exp(gamma_L * log_obs_factor) / 
                    (integral_term ** (gamma_L/(1.0+gamma_L))))
        else:
            sys.exit("ERROR: You Either specified a losstype that does not "+
                     "exist or specified a losstype without specifying its " +
                     "hyperparameter. Implemented losstypes are " + 
                     "'standard', 'beta', 'gamma'.")
        return np.mean(log_lkl, 0)
    
    """UNIVERSAL. 
        Draw from variational posterior q of size K
    """
    def draw_samples(q, K):
        return npr.randn(K, len(q[ 'm' ])) * np.sqrt(q[ 'v' ]) + q[ 'm' ]
    
    """A-APPROX.
        Helper function inside the energy / objective function
        Ported directly from their public code release.
    """
    def log_likelihood_factor_BBalpha(samples_q, v_noise, X, y):
        outputs = predict(samples_q, X)
        return -0.5 * np.log(2 * math.pi * v_noise) - 0.5 * (np.tile(y, (1, samples_q.shape[ 0 ])) - outputs)**2 / v_noise
    
    """AR/AB-APPROX.
        Helper function inside the energy / objective function
        Ported directly from AR-Approx public code release.
    """
    def log_q(samples_q, q):
        log_q = -0.5 * np.log(2 * math.pi * q[ 'v' ]) - 0.5 * (samples_q - q[ 'm' ]) **2 / q[ 'v' ]
        return np.sum(log_q, 1)
    
    """A-APPROX.
        Helper function inside the energy / objective function
        Ported directly from their public code release.
    """
    def log_normalizer(q): 
        return np.sum(0.5 * np.log(q[ 'v' ] * 2 * math.pi) + 0.5 * q[ 'm' ]**2 / q[ 'v' ])
    
    """AR/AB-APPROX.
        Helper function inside the energy / objective function
        Ported directly from AR-Approx public code release.
    """
    def log_prior(samples_q, v_prior, m_prior):
        log_p0 = -0.5 * np.log(2 * math.pi * v_prior) - 0.5 * (samples_q - m_prior) **2 / v_prior
        return np.sum(log_p0, 1)   
    
    """A-APPROX.
        Helper function inside the energy / objective function
        Ported directly from their public code release.
    """
    def log_Z_prior(v_prior, m_prior):
        return N * (
                (0.5 * np.log(v_prior * 2 * math.pi)) + 0.5 * m_prior**2 / v_prior)
    
    """A-APPROX.
        Helper function inside the energy / objective function
        Ported directly from their public code release.
    """
    def log_Z_likelihood(q, f_hat, v_noise, X, y, K):
        samples = draw_samples(q, K)
        log_f_hat = np.sum(-0.5 / f_hat[ 'v' ] * samples**2 + f_hat[ 'm' ] / f_hat[ 'v' ] * samples, 1)
        log_factor_value = alpha * (log_likelihood_factor_BBalpha(samples, v_noise, X, y) - log_f_hat)
        return np.sum(logsumexp(log_factor_value, 1) + np.log(1.0 / K))
    
    """A-APPROX.
        Helper function inside the energy / objective function
        Ported directly from their public code release.
    """
    def get_parameters_f_hat(q, v_prior, m_prior, N):
        v = 1.0 / (1.0 / N * (1.0 / q[ 'v' ] - 1.0 / v_prior))
        m = (v / N) * (q['m'] / q['v'] - m_prior/v_prior)  #1.0 / N * q[ 'm' ] / q[ 'v' ] * v
        return { 'm': m, 'v': v }
    
    
    """UNIVERSAL.
        Just an unpacking function for the parameters
    """
    def get_parameters_q(w, v_prior, scale = 1.0):
        v = 1.0 / (scale * np.exp(-parser.get(w, 'log_variance'))[ :, 0 ] + 1.0 / v_prior)
        m = scale * parser.get(w, 'mean')[ :, 0 ] * np.exp(-parser.get(w, 'log_variance')[ :, 0 ]) * v
        return { 'm': m, 'v': v }
    
    """GVI.
        Implements the alpha-renyi prior regularizer in closed form (between
        fully factorized normals)
    """
    def prior_regularizer(q, v_prior, m_prior, alpha):
        """NOTE: New function returning the prior reg"""
        
        """Get the normalizing constants & the regularizer"""
        logZpi =  0.5 * np.log(v_prior) + 0.5 * m_prior ** 2 / v_prior
        logZq = (0.5 * np.log(q['v']) + 0.5 * q['m']**2 / q['v'])
        
        new_var = 1.0 / (alpha / q['v'] + (1.0-alpha) /v_prior)
        new_mean_ = (alpha * q['m']/q['v'] + (1-alpha) * m_prior/v_prior) 
        logZnew = (0.5 * np.log(new_var) + 0.5 * (new_mean_**2) * new_var)
        
        log_reg = np.sum(logZnew - alpha*logZq - (1-alpha)*logZpi)
        
        """return it"""
        return log_reg

    """UNIVERSAL.
        Implements the objective L(q|x, D, \ell_n) 
    """
    def energy(w, X, y, v_prior, m_prior, K, N, alpha):
        """Extract parameters"""
        q = get_parameters_q(w, v_prior)
        v_noise = np.exp(parser.get(w, 'log_v_noise')[ 0, 0 ])
        
        """Note: A-approx computes its own log_factor value inside the helper
        function log_Z_likelihood, so we can shave of some computation time"""
        if Dtype != "A-approx":
            samples_q = draw_samples(q, K)
            log_factor_value = 1.0 * N * log_likelihood_factor(samples_q, v_noise, X, y)
        
        if Dtype == "KL":
            """I.e., standard VI"""
            KL = np.sum(-0.5 * np.log(2 * math.pi * v_prior) - 0.5 * ((q[ 'm' ]-m_prior)**2 + q[ 'v' ]) / v_prior) - \
                np.sum(-0.5 * np.log(2 * math.pi * q[ 'v' ] * np.exp(1)))
            vfe = -(np.mean(log_factor_value) + KL)
            
        elif Dtype == "AR-approx":
            """NOTE: Needs modification to be GVI"""
            logp0 = log_prior(samples_q, v_prior, m_prior)
            logq = log_q(samples_q, q)
            logF = logp0 + log_factor_value - logq
            logF = (1 - alpha) * logF
            vfe = -(logsumexp(logF) - np.log(K))            
            vfe = vfe / (1 - alpha)
            
        elif Dtype == "AB-approx":
            logp0 = log_prior(samples_q, v_prior, m_prior)
            logq = log_q(samples_q, q)
            part1 = (alpha + beta_D) * (log_factor_value  + logp0) - logq
            part2 = (alpha + beta_D -1 ) * logq
            part3 = (beta_D * (log_factor_value  + logp0) + (alpha - 1) * logq)
            vfe = ( (1.0 / (alpha * (alpha + beta_D))) * (logsumexp(part1) - np.log(K)) 
                    + (1.0 / (beta_D * (alpha + beta_D))) * (logsumexp(part2) - np.log(K)) 
                    - (1.0 / (alpha * beta_D)) * (logsumexp(part3) - np.log(K)))
            
        elif Dtype == "A-approx":
            f_hat = get_parameters_f_hat(q, v_prior, m_prior, N) 
            vfe =  -log_normalizer(q) - 1.0 * N / X.shape[ 0 ] / alpha * log_Z_likelihood(q, f_hat, 
                              v_noise, X, y, K) + log_Z_prior(v_prior,m_prior)
            
        elif Dtype == "AR":
            prior_reg = (1/(alpha*(alpha-1))) * prior_regularizer(q,v_prior, m_prior,alpha)
            vfe = -np.mean(log_factor_value) + prior_reg
        
        #NOTE: While this should work, this is the alpha-divergence regularizer, which 
        #       overconcentrates substantially. We refer to the appendix of our 
        #       paper for some visuals on this phenomenon. The performance from this
        #       divergence should be expected to be much worse than that for the
        #       Alpha-renyi as Uncertainty Quantifier
        elif Dtype == "A":
            prior_reg = (1/(alpha*(alpha-1))) * (
                    np.exp(prior_regularizer(q,v_prior, m_prior,alpha))-1)
            vfe = -np.mean(log_factor_value) + prior_reg
            
        return vfe

    """UNIVERSAL.
        Implements the test RMSE and test log likelihood. 
    """
    def get_error_and_ll(w, v_prior, X, y, K, location, scale):
        v_noise = np.exp(parser.get(w, 'log_v_noise')[ 0, 0 ]) * scale**2
        q = get_parameters_q(w, v_prior)
        samples_q = draw_samples(q, K)
        outputs = predict(samples_q, X) * scale + location
        log_factor = -0.5 * np.log(2 * math.pi * v_noise) - 0.5 * (np.tile(y, (1, K)) - np.array(outputs))**2 / v_noise
        ll = np.mean(logsumexp(log_factor - np.log(K), 1))
        error = np.sqrt(np.mean((y - np.mean(outputs, 1, keepdims = True))**2))
        return error, ll
    
    """UNIVERSAL.
        Implements a way to simulate from and validate the predictive distribution.
        This is how the figures were produced that show F-VI over-concentrates
    """
    def get_predictive_distribution(w, v_prior, X, K, location, scale):
        """Approximate the predictive distro by returning K samples from 
        q(y_i|X_i) for i=1,2, ..."""
        
        #Get predictions
        q = get_parameters_q(w, v_prior)
        samples_q = draw_samples(q, K)
        outputs_raw = predict(samples_q, X) * scale + location
        
        #Add the noise v_noise
        v_noise = np.exp(parser.get(w, 'log_v_noise')[ 0, 0 ]) * scale**2
        outputs = outputs_raw + np.random.normal(size=outputs_raw.shape) * np.sqrt(v_noise)
        
        return outputs, outputs_raw, v_noise
    

    return w, energy, get_error_and_ll, get_predictive_distribution



