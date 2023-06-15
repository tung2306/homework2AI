import numpy as np



import numpy as np

def mapFeature(X1, X2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix or a one-dimensional array
        X2 is an n-by-1 column matrix or a one-dimensional array
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    degree = 6  # Maximum degree of polynomial terms
    
    # Reshape X1 and X2
    X1 = np.reshape(X1, (X1.shape[0], 1))
    X2 = np.reshape(X2, (X2.shape[0], 1))
    
    # Initialize an empty feature matrix
    mapped_features = np.ones((X1.shape[0], 1))
    
    for i in range(1, degree + 1):
        for j in range(i + 1):
            # Compute the polynomial term
            term = (X1 ** (i - j)) * (X2 ** j)
            # Append the term to the feature matrix
            mapped_features = np.hstack((mapped_features, term))
    
    return mapped_features


