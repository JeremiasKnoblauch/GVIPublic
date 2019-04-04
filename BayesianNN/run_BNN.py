
import time
import numpy as np
import sys

from GVI_BNN import fit_q



"""Function serves three purposes: 
    (1) Prepares the accurate split for index i
    (2) Calls 'fit_q' for split i, which does the inference
    (3) Extracts and returns test performancee metrics to 'main'.
"""
def get_test_error(X, y, i, dataset, alpha = 0.0, learning_rate=0.001, 
                   epochs=500, Dtype = "AR", losstype = None, 
                   beta_L = None, gamma_L = None, beta_D = -0.3, 
                   store_predictive_distribution=False):
    
    """STEP (1): Prepare the i-th split"""
    
    # We fix the random seed
    np.random.seed(1)
    
    # We load the indices of the training and test sets
    path = 'data/' + dataset + '/'
    index_train = np.loadtxt(path + "index_train_{}.txt".format(i))
    index_test = np.loadtxt(path + "index_test_{}.txt".format(i))
    
    # load training and test data
    X_train = X[ [int(e) for e in index_train.tolist()], ]
    y_train = y[ [int(e) for e in index_train.tolist()] ]
    X_test = X[ [int(e) for e in index_test.tolist()], ]
    y_test = y[ [int(e) for e in index_test.tolist()] ]

    # We normalize the features
    std_X_train = np.std(X_train, 0)
    std_X_train[ std_X_train == 0 ] = 1
    mean_X_train = np.mean(X_train, 0)
    X_train = (X_train - mean_X_train) / std_X_train
    X_test = (X_test - mean_X_train) / std_X_train
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)
    y_train = (y_train - mean_y_train) / std_y_train
    y_train = np.array(y_train, ndmin = 2).reshape((-1, 1))
    y_test = np.array(y_test, ndmin = 2).reshape((-1, 1))


    # Hard-coded inference settings used throughout all runs in paper
    batch_size = 32
    epochs = 500
    learning_rate = 0.001
    K = 100
    hidden_layer_size = 50
    m_prior = 0.0
    v_prior = 1.0
    
    
    """STEP (2): Call 'fit_q' for split i, which does the inference on i"""
    #Call to the class/objects dooiing all the heavy lifting (GVI_BNN)
    start_time = time.time()
    w, v_prior, get_error_and_ll, get_predictive_distribution = fit_q(X_train, y_train, hidden_layer_size, 
        batch_size, epochs, K, mean_y_train, std_y_train,
        alpha, learning_rate, v_prior, Dtype, losstype, 
        beta_L, gamma_L, beta_D,
        m_prior
        )
    running_time = time.time() - start_time

    # We obtain the test RMSE and the test ll

    
    """STEP (3): Extract and return test performancee metrics for split i"""
    
    error, ll = get_error_and_ll(w, v_prior, X_test, y_test, K, mean_y_train, std_y_train)
        
    #If we want to plot the predictives, get an approx to the pred distro
    predictive = None
    if store_predictive_distribution:
        predictive = get_predictive_distribution(w, v_prior, X_test, K*10, 
                                        mean_y_train, std_y_train)
        

    return -ll, error, running_time, predictive, y_test


"""Function serves three purposes: 
    (1) Reads in the (full) data as specified via 'dataset'
    (2) Calls 'get_test_error' for each split, which will build the BNN model
        for that split and return the test errors for the 10% hold out data
    (3) Store the results from step (2) on the hard drive.
"""
def main(split_num, dataset, alpha, learning_rate, epochs, Dtype, losstype, 
         beta_L, gamma_L, beta_D,
         subfolder="", store_predictive_distribution=False):
    
    
    """STEP (1): Read in the full data set togetheer with the indices for 
                 inputs x and outputs y"""
    # We load the data. Some of it is stored as .csv rather than .txt
    special_treatment = ['power', 'kin8mn', 'energy']
    
    datapath = 'data/' + dataset + '/'
    if dataset not in special_treatment:
        data = np.loadtxt(datapath + 'data.txt')
    elif dataset == 'power':
        data = np.loadtxt(datapath +'data.csv', skiprows=1, delimiter =",")
    elif dataset == 'kin8mn':
        data = np.loadtxt(datapath + 'data.csv', delimiter =",")
    elif dataset == 'energy':
        data = np.loadtxt(datapath + 'data.csv', skiprows=1, delimiter =",")
        
    # We extract indices of the data for y (target) and x (features) 
    index_features = np.loadtxt(datapath + 'index_features.txt')
    index_target = np.loadtxt(datapath + 'index_target.txt')
    
    #Apply these indices to x and y.
    if type(index_features.tolist()) == list:
        indfeat = [int(e) for e in index_features.tolist()]
    else:
        indfeat = int(index_features.tolist())
    
    if type(index_target.tolist()) == list:
        indtarg = [int(e) for e in index_target.tolist()]
    else:
        indtarg = int(index_target.tolist())

    X = data[ : , indfeat ]
    y = data[ : , indtarg ]


    """STEP (2.1): Get all splits you want to run"""
                 
    #You can specify a selection of indices for the data splits, a single split
    #or None (in which case all splits are run)
    if split_num is None:
        n_splits = 50
        splits = range(n_splits)
    elif type(split_num) == int:
        splits = np.array([split_num])
    elif type(split_num) == list:
        splits = np.array(split_num)
    
    #Loop over all selecteed splits
    savepath = 'results/' + subfolder + '/'
    for i in splits: 
        
        """STEP (2.2): For each split, call 'get_test_error"""
        
        print('split', i+1)
        neg_test_ll, test_error, running_time, predictive, y_test = get_test_error(
                                    X, y, i+1, dataset, 
                                    alpha, learning_rate, epochs, Dtype, losstype, 
                                    beta_L, gamma_L, 
                                    beta_D,
                                    store_predictive_distribution)
        
        
        """STEP (2.3): For each split, save the test errors and other 
                        quantities of interest (like the predictive distros)
        """
        
        file_string = (savepath + dataset + "_alpha" + str(alpha) + 
                    "_lr" + str(learning_rate) + "_epochs" + str(epochs) +
                    "_Dtype" + Dtype + "losstype_" + losstype +                     
                    "_split" + str(split_num))
    
        with open(file_string + "_test_ll.txt", 'a') as f:
                f.write(repr(neg_test_ll) + '\n')
        with open(file_string + "_test_error.txt", 'a') as f:
                f.write(repr(test_error) + '\n')
        with open(file_string + "_test_time.txt", 'a') as f:
                f.write(repr(running_time) + '\n')
        
        if store_predictive_distribution:
            name4 = file_string + "_sigma2.txt"
            with open(name4, 'a') as f:
                f.write(repr(predictive[2]) + '\n')

if __name__ == '__main__':
    """Refer to the README file to navigate & run the desired settings. 
    
    Important notes:
        - F-VI methods CANNOT be combined with robust losses 
            (i.e., losstype = 'standard' required for any F-VI Dtype spec)
        - If Dtype = 'AB-approx' (i.e., F-VI with F=scaled alpha-beta-div), 
            the loss_hyperparam argument becomes the beta argument of F.
        - You always have to specify alpha/the loss hyperparameter, even if
            your setting does not need them. We recommend you set them to -99
            in this case. If you do not, the program will set them to -99 
            internally and give you a warning (which won't interrupt execution)
    
    """
    
    
    """BLOCK I: Data & inference method"""
    dataset = str(sys.argv[1]) #boston, concrete, energy, kin8mn, naval, power, wine, yacht
    Dtype = str(sys.argv[2])   #Which type of Q-constrained posterior do you want?
                               #refer to readme for detailed description. Available choices:
                               # 'KL' (= standard VI uncertainty quantifier)
                               # 'AR' (= Renyi-alpha div uncertainty quantifier)
                               # 'A-approx' (=alpha-divergence black box as in Hernandez-Lobato et al. '16)
                               # 'AR-approx' (=alpha-renyi divergence F-VI as in Li & Turner '16)
                               # 'AB-approx' (=(scaled) alpha-beta div. as in Regli & Silva, '18)
    alpha = float(sys.argv[3]) #Meaning depends on Dtype argument.
                               #alpha-parameter for GVI with D = Renyi-alpha-div (Dtype = 'AR')
                               #alpha-parameter for F-VI with F = Renyi-alpha-div (Dtype = 'AR-approx')
                               #alpha-parameter for F-VI with F = alpha-div (Dtype = 'A-approx')
                               #alpha-parameter for F-VI with F = alpha-beta-div (Dtype = 'AB-approx')
    losstype = str(sys.argv[4])     #Type of loss used for (log-lklh, beta, gamma)
                                    #admissible choices: 
                                    #   'b' for beta-loss
                                    #   'g' for beta-loss
                                    #   'standard' negative log-loss
    loss_hyperparam = float(sys.argv[5])  #hyperparameter of the loss
                                          #beta if losstype = 'b'
                                          #gamma if losstype = 'g'
                                          #anything/nothing if losstype = 'standard'
                                          #NOTE: If Dype='AB-approx', this becomes
                                          #      beta-argument of AB-div.
    
    
    
    """BLOCK II: Optimization hyperparameters (kept fixed in the paper) + 
                  some additional options for convenience."""
    
    learning_rate = 0.001
    epochs = 500
    split_num = None          #could also be a list of splits/an single index (integer)
    subfolder = ""            #in which subfolder of 'results' to store this run
    store_predictive_distribution = False #allows you to store predictive distributions
                                          #for each split you run
    
    
    """Sanity check I: Make sure that F-VI methods are not specified together 
                        with robust losses. Forcibly change to standard loss 
                        and notify user"""
    
    print('\n')
    FVI_bool = (Dtype == 'A-approx' or Dtype == 'AR-approx' or Dtype == 'AB-approx')
    if losstype != 'standard' and FVI_bool:
        print("WARNING! You specified a robust loss together with an F-VI method! " + 
              "The program forced your loss to be the standard negative log likelihood.")
        print('\n')
        losstype = 'standard'                       
                        
    """Sanity check II: Make sure that if Dtype=AB-approx, then a loss_hyperparam
                        is specified. Forcibly set beta = -0.25 if not specified
                        (overall best value in simulations of Regli & Silva '18)
                        and notify user"""
    
    if Dtype == 'AB-approx' and loss_hyperparam is None:
        print("WARNING! You specified Dtype=AB-approx, but your loss_hyperparam " + 
              "was left unspecified. If you specify AB-approx, loss_hyperparam " +
              "takes the role of beta (B). The program forces beta (B) = -0.25")
        print('\n')
        loss_hyperparam = -0.25
    
    """Sanity check III: Make sure Dtype is admissible"""
    Dtypes = ["KL", "AR", "A", "A-approx", "AR-approx", "AB-approx"]
    if Dtype not in Dtypes:
        print("ERROR! You specified an invalid argument for Dtype! Has to be one of ")
        print(Dtypes)
        print("Execution was aborted.")
        sys.exit(0)
    
    """Sanity check IV: Make sure losstype is admissible"""
    losstypes = ["b", "g", "standard"]
    if losstype not in losstypes:
        print("ERROR! You specified an invalid argument for losstype! Has to be one of ")
        print(losstypes)
        print("Execution was aborted.")
        sys.exit(0)
    

    """Recasting: Cast beta_L or gamma_L as loss_hyperparam if needed. Similarly,
                  cast beta_D as hyperparam if Dtype = 'AB-approx'."""

    beta_L, gamma_L, beta_D = None, None, None
    if losstype == 'b':
        #NOTE: We use a different parameterization for the code than inside the paper.
        #       In particular, beta_code = beta_paper - 1. 
        beta_L = loss_hyperparam - 1 
    elif losstype == 'g':
        #NOTE: We use a different parameterization for the code than inside the paper.
        #       In particular, gamma_code = gamma_paper - 1. 
        gamma_L = loss_hyperparam - 1
    elif losstype == 'standard' and Dtype != "AB-approx" and loss_hyperparam != -99:
        #NOTE: Whenever it is not being used, we set the loss hyperparam to -99
        print("WARNING! Your loss parameter was not set to -99 in spite of not " +
              "being required/used for your setting. The program forced your loss " + 
              "parameter to be -99!" )
        print('\n')
        loss_hyperparam = -99

    if Dtype == 'AB-approx':
        print("You selected Dtype=AB-approx. Your loss parameter will be used " + 
              "to set the beta (B) argument")
        print('\n')
        beta_D = loss_hyperparam 
        
    if Dtype == 'KL' and alpha != -99:
        print("WARNING! You selected Dtype=KL, but alpha was not set to -99 in " +
              "spite of not being required/used for your setting. The program " +
              "forced your alpha parameter too be -99!")
        print('\n')
        alpha = -99
    
    if split_num is None:
        n_splits = 50
        splits = range(n_splits)
    else:
        splits = split_num
        
    """Print summary statement of setting that is being run"""
    print("{0:18}|{1:18}|{2:18}|{3:18}|{4:18}".format(
            "data set", "inference method", "alpha", "losstype", "loss parameter"))
    print("{0:18}|{1:18}|{2:18}|{3:18}|{4:18}".format(
            dataset, Dtype, alpha, losstype, loss_hyperparam))
    print('\n')
    print("Splits to be computed: ", splits)
    print('\n')


    """Run setting"""
    main(split_num, dataset, alpha, learning_rate, epochs, Dtype, losstype, 
         beta_L, gamma_L, beta_D, 
         subfolder, store_predictive_distribution)
    
    
