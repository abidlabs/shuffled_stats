from __future__ import print_function
import numpy as np
from scipy.optimize import minimize
import math

def linregress(x, y, estimator=None, groups=None, verbose=False, l1=0, l2=0, n_starts=None, default_numerical=False, options=None, noise=0):
    #if x is flattened, then reshape it
    if len(x.shape)==1:
        x = x.reshape(x.shape[0], 1)
    if len(x.shape)>2:
        raise ValueError('x must be a 1-D or 2-D array')
    if len(y.shape)>1:
        raise ValueError('y must be a 1-D array')
    
    ESTIMATORS = ['LS', 'SM', 'SM_analytical', 'P1', 'P2', 'EMD', 'SMALL_D', 'KS']
    if not(estimator is None) and not(estimator in ESTIMATORS):
        raise ValueError('estimator, if provided, must be one of: ' + str(ESTIMATORS))
    n, d = x.shape
    
    #get number of groups
    if (groups is None):
        n_groups = 1
    else:
        n_groups = len(np.unique(groups))
    
    if (options is None):
        #default arguments
        options = [int(d/n_groups)]
    
    #get number of starts
    if (n_starts is None):
        n_starts = 2*d**2

    #if estimator is not specified, choose it here:
    if (estimator is None):
        estimator = choose_estimator(n, d, n_groups)
    if (verbose):
        print("Using",estimator,"estimator")
        print("with",n_starts, "initializations (for non-analytical estimators)")
        print("n=",n, "d=",d,"C=",n_groups)
        print("Options:",str(options))
    
    #Handle the case where there is only one set of measurements
    #if (n_groups == 1):
    if (estimator=='SM_analytical'):
        w = SM_analytical(x, y, groups, noise=noise)
    elif (estimator=='SM' and ((d<=2 and n_groups==1) or (d<=n_groups)) and not(default_numerical)):
        w = SM_analytical(x, y, groups)            
    elif (estimator=='P1'):
        w = minimize_weights_by_projections(x, y, 1, n_starts, l1, l2, groups)
    elif (estimator=='P2'):
        w = minimize_weights_by_projections(x, y, 2, n_starts, l1, l2, groups)
    else:
        w = minimize_weights_by_gradient_descent(x, y, estimator, n_starts, l1, l2, groups, args=options)
    return w


def load_dataset_in_clusters(filename, normalize=True, n_clusters = 1):
    data = np.genfromtxt(filename, delimiter=',')
    n,d = data.shape
    
    cluster_vector = np.tile(range(n_clusters), n//n_clusters)
    cluster_vector = np.concatenate((cluster_vector, range(n%n_clusters)))
    np.random.shuffle(cluster_vector)
    
    #normalize
    if (normalize):
        data = (data-np.min(data, axis=0))
        data = data/(np.max(data, axis=0)+0.01)
    
    labels = data[:,-1].copy()
    features = data[:,:].copy()
    features[:,-1] = 1 #add a bias column
    
    return features, labels, cluster_vector

def minimize_weights_by_projections(x, y, proj_dim, n_starts, l1, l2, groups):
    if not(proj_dim==1 or proj_dim==2):
        raise ValueError("The projection dimension must be 1 or 2")
    _, d = x.shape
    if (d<=proj_dim):
        #print("Warning: the projection dimension is <= the dimensionality of the dataset, so reducing to LS estimator")
        return minimize_weights_by_gradient_descent(x, y, 'LS', n_starts, l1, l2, groups, args=[d])
    #store the cost associated with each projection matrix
    weights = np.zeros((n_starts, d))
    costs = np.full(n_starts, float("Inf"))
    for i in range(n_starts):
        p0 = np.random.normal(0, 1, d-proj_dim)
        p_rand = np.random.normal(0, 1, (d, proj_dim))
        store_values([x, y, proj_dim, l1, l2, groups, p_rand])
        res = minimize(evaluate_cost_of_projection, p0)
        #store the weights and cost
        #p = p_rand.dot(res.x).reshape(d, proj_dim)
        p = p_rand.copy(); p[0:d-proj_dim,0] = res.x
        x_tilde = x.dot(p)
        w_tilde = SM_analytical(x_tilde, y)
        w = p.dot(w_tilde).flatten()
        #print(w_tilde, w)
        weights[i, :] = w
        costs[i] = res.fun
    #print(costs)
    return weights[np.argmin(costs)]        
    
def evaluate_cost_of_projection(p_):
    x, y, proj_dim, l1, l2, groups, p_rand = retrieve_values()
    _, d = x.shape
    #p = p_rand.dot(p_).reshape(d, proj_dim)
    #print(p_)
    p = p_rand.copy(); p[0:d-proj_dim,0] = p_
    if groups is None:
        x_tilde = x.dot(p)
        w_tilde = SM_analytical(x_tilde, y)
        w = p.dot(w_tilde).flatten()
        y_ = x.dot(w)
        cost = LS(y, y_) + l2*np.mean(np.square(w)) + l1*np.mean(np.abs(w))
    else:
        clusters, indices = np.unique(groups, return_inverse=True)
        cost = 0
        for c in clusters:
            x_h = x[np.where(indices==c)]
            y_h = y[np.where(indices==c)]
            x_tilde = x_h.dot(p)
            w_tilde = SM_analytical(x_tilde, y_h)
            w = p.dot(w_tilde).flatten()
            y_ = x_h.dot(w).flatten()
            cost += LS(y_h, y_)
        cost += l2*np.mean(np.square(w)) + l1*np.mean(np.abs(w))
    #print(cost)
    return cost
    
def minimize_weights_by_gradient_descent(x, y, estimator, n_starts, l1, l2, groups, args):
    #so they can be accessed in the cost function
    store_values([x, y, estimator, l1, l2, groups, args])
    _, d = x.shape
    #store the cost associated with each weight
    weights = np.zeros((n_starts, d))
    costs = np.full(n_starts, float("Inf"))
    for i in range(n_starts):
        w0 = np.random.normal(0, 1, d)
        res = minimize(evaluate_cost, w0)
        weights[i, :] = res.x
        costs[i] = res.fun
    return weights[np.argmin(costs)]    

def evaluate_cost(w):
    import math
    x, y, estimator, l1, l2, groups, args = retrieve_values()
    func = globals()[estimator]
    if groups is None:
        y_ = x.dot(w).flatten()
        cost = func(y, y_, args) + l2*np.mean(np.square(w)) + l1*np.mean(np.abs(w))
    else:
        clusters, indices = np.unique(groups, return_inverse=True)
        cost = 0
        for c in clusters:
            x_h = x[np.where(indices==c)]
            y_ = x_h.dot(w).flatten()
            y_h = y[np.where(indices==c)]
            cost += func(y_h, y_, args)
        cost += l2*np.mean(np.square(w)) + l1*np.mean(np.abs(w))
    #print(w, cost)
    return cost

#helper methods to store global values to share across functions (used to simplify function calls to scipy.minimize module)    
def store_values(values):
    global cached_values 
    cached_values = values

def retrieve_values():
    global cached_values
    return cached_values

#helper method that contains the rules for deciding which estimator to use
def choose_estimator(n, d, n_groups=1):
    #if there aren't groups, then just choose on the basis of n and d
    if (n_groups==1):
        if (d==1 or d==2):
            estimator = 'SM'
        if (d>2 and n>=1000):
            estimator = 'P1'
        if (d>2 and n<=1000):
            estimator = 'LS'
    else:
        if (n_groups >= 3*d):
            estimator = 'SM'
        else:
            estimator = 'LS'
    return estimator

#estimators, args is unused parameter (may be used in the future)
def LS(y, y_, args=None):
    if len(y.shape)>1 or len(y_.shape)>1:
        raise ValueError('Arrays must be 1-D for comparison')
    y = np.sort(y)
    y_ = np.sort(y_)
    n = len(y)
    return 1.0/n * np.sum(np.square(y_ - y))

#TODO: include higher moments
def SM(y, y_, args=None):
    import math
    d = args[0]
    cost = 0
    y1 = y.copy()
    y2 = y_.copy()
    for i in range(1, d+2):
        cost += 1.0/math.factorial(i) * np.square(np.mean(y1) - np.mean(y2))
        y1 = np.multiply(y,y1)
        y2 = np.multiply(y_,y2)
    return cost

#TODO: include higher dimensions 
def SM_analytical(x, y, groups=None, noise=0):
    n, d = x.shape
    n_clusters = 0 #by default, unless a grouping vector is provided
    if not(groups is None):
        clusters, indices = np.unique(groups, return_inverse=True)
        n_clusters = len(clusters)
    if (n_clusters >= d and n_clusters>1):
        mean_x = np.zeros((n_clusters, d))
        mean_y = np.zeros((n_clusters, 1))
        for i, c in enumerate(clusters):
            x_h = x[np.where(indices==c)]
            y_h = y[np.where(indices==c)]
            mean_x[i] = np.mean(x_h, axis=0)
            mean_y[i] = np.mean(y_h)
        w = np.linalg.lstsq(mean_x, mean_y)[0]
        return w.flatten()
    elif (d==1):
        return np.mean(y)/np.mean(x)
    elif (d==2):
        a, b, c, Ex1, Ex2, Ey = get_coefficients(x,y, noise)
        w2, v2 = solve_quadratic(a, b, c)
        w1 = solve_w1(w2, Ex1, Ex2, Ey)
        v1 = solve_w1(v2, Ex1, Ex2, Ey)
        vs = np.real(np.array([v1, v2]))
        ws = np.real(np.array([w1, w2]))
        y1 = np.dot(x, vs)
        y2 = np.dot(x, ws)
        if LS(y1,y)<LS(y2,y):
            return vs
        else:
            return ws        

    raise ValueError("Analytical solution for SM estimator only exists for d=1,2 (without clusters) or d >= C (with multiple clusters)")
        
def get_coefficients(X,y, noise=0):
    THRESH = 1e-6
        
    n = X.shape[0]
    Ex1 = float(np.mean(X,axis=0)[0])
    Ex2 = float(np.mean(X,axis=0)[1])
    x12 = X[:,0].dot(X[:,0])/float(n)
    x22 = X[:,1].dot(X[:,1])/float(n)
    x1x2 = X[:,0].dot(X[:,1])/float(n)
    Ey = float(np.mean(y))
    y2 = float(np.mean(np.square(y)) - noise**2)
    
    if (abs(Ex1) < THRESH or abs(Ex2) < THRESH):
        print("Warning: the analytical SM estimator may produce very large or small weights. Please confirm solution with another estimator.")
    if (abs(Ex1) < THRESH): #to avoid divide-by-zero errors
        Ex1 = THRESH
    if (abs(Ex2) < THRESH): #to avoid divide-by-zero errors
        Ex2 = THRESH
    
    a = ((Ex2/Ex1)**2)*x12-(Ex2/Ex1)*(2*x1x2)+x22
    b = (2*x1x2)*(Ey/Ex1) - 2*(Ey/Ex1)*(Ex2/Ex1)*x12
    c = (Ey/Ex1)**2*x12-y2
    return a,b,c, Ex1, Ex2, Ey

#Helper method to solves the quadratic equation
def solve_quadratic(a,b,c,flag_complex=False):
    from cmath import sqrt
    THRESH = 1e-6
    a = float(a) + 0j;
    b = float(b) + 0j;
    c = float(c) + 0j;
    if (abs(a) < THRESH): #to avoid divide-by-zero errors
        a = THRESH
    if ((b * b) - 4 * a * c)<0 and flag_complex==True:
        print("**Warning: Complex weights..")
    discRoot = sqrt((b * b) - 4 * a * c) # first pass
    root1 = (-b + discRoot) / (2 * a) # solving positive
    root2 = (-b - discRoot) / (2 * a) # solving negative
    
    #strip the complex part if its not a complex number
    return (root1, root2) if abs(root1.imag)>THRESH else (root1.real, root2.real)

#Helper method that solves for one root given the other
def solve_w1(w2, Ex1, Ex2, Ey):
    THRESH = 1e-6
    w1 = (Ey/Ex1) - w2*(Ex2/Ex1) + 0j
    return w1 if abs(w1.imag)>THRESH else w1.real

def error_in_weights(w0, w1):
    w0 = np.array(w0)
    w1 = np.array(w1)
    w0 = w0.flatten()
    w1 = w1.flatten()
    err = np.linalg.norm(w0-w1)/np.linalg.norm(w0)
    return err
    
def generate_dataset(dim=2, n=100, noise=0, dist='normal', mean=0, var=1, weights=None, bias=False, n_groups=None):
    """generates data of a given dimension and distribution with given parameters
    weights -- if you would like to set the weight matrix, provide it here
    """    
    if (dist=='normal'):
        X = np.random.normal(mean,var, size=[n, dim]);
    elif (dist=='half-normal-uniform'):      
        X1 = np.random.rand(n, dim//2)-mean*0.5
        X2 = np.random.normal(mean, var, size=[n, dim//2])
        X = np.concatenate((X1, X2), axis=1)
    elif (dist=='uniform'):
        X = var*np.random.rand(n, dim)+mean
    elif (dist=='2normals'):
        X1 = np.random.normal(-mean,var, size=[n/2, dim])
        X2 = np.random.normal(mean,var, size=[n-n/2, dim]);
        X = np.concatenate((X1, X2), axis=0)
    elif (dist=='exponential'):
        X = np.random.exponential(var, size=[n, dim]);
    else:
        raise NameError('Invalid distribution: ' + str(dist))
    if (bias):
        X[:,0] = 1 #set the first column to be all 1s for the "intercept" term
    if (weights is None):
        weights = np.random.normal(0,1,size=[dim]); #Actual weights
    else:
        weights = np.array(weights)
        weights = weights.flatten()
    y = np.dot(X,weights) + noise*np.random.normal(0,1,size=[n]); #Ordered labels
    if (n_groups is None):
        return X, y, weights
    else:
        groups_vector = np.tile(range(n_groups), n//n_groups)
        groups_vector = np.concatenate((groups_vector, range(n%n_groups)))
        np.random.shuffle(groups_vector)
        return X, y, weights, groups_vector

#TODO: write an SM_analytical for clusters?


def EMD(y, y_, args=None):
    import math

    if len(y.shape)>1 or len(y_.shape)>1:
        raise ValueError('Arrays must be 1-D for comparison')

    y = np.sort(y)
    x = np.sort(y_) #so I don't have to change anything in the code below
    mn = min([min(x),min(y)])
    
    dist = 0 #a running sum of the EMD
    cdf_y = 0; cdf_x = 0 #initialize the CDFs
    pos = mn; pos_prev = mn; #pointers to help compute the CDFs
    x = np.append(x, math.inf); y = np.append(y, math.inf) #guarantees all samples will be run through
    ix = 0; iy = 0; #points to indices within the lists x and y
    
    #terminate when the Infinity elements are reached in both lists
    while (ix < len(x)-1 or iy < len(y)-1): 
        prev_pos = pos #update the previous position
        if x[ix]<y[iy]:
            pos = x[ix];
            ix += 1; 
            dist += (pos-prev_pos)*abs(cdf_x - cdf_y)
            cdf_x += 1.0/(len(x)-1);
        else:
            pos = y[iy];
            iy += 1; 
            dist += (pos-prev_pos)*abs(cdf_x - cdf_y)
            cdf_y += 1.0/(len(y)-1);
    
    return dist**2

def SMALL_D(y, y_, args=None):
    if len(y.shape)>1 or len(y_.shape)>1:
        raise ValueError('Arrays must be 1-D for comparison')
    d = args[0]
    y = np.sort(y)[:d]
    y_ = np.sort(y_)[:d]
    n = len(y)
    return 1.0/n * np.sum(np.square(y_ - y))

def KS(y, y_, args=None):
    import math

    if len(y.shape)>1 or len(y_.shape)>1:
        raise ValueError('Arrays must be 1-D for comparison')

    y = np.sort(y)
    x = np.sort(y_) #so I don't have to change anything in the code below
    mn = min([min(x),min(y)])
    
    dist = 0 #a running sum of the EMD
    cdf_y = 0; cdf_x = 0 #initialize the CDFs
    pos = mn; pos_prev = mn; #pointers to help compute the CDFs
    x = np.append(x, math.inf); y = np.append(y, math.inf) #guarantees all samples will be run through
    ix = 0; iy = 0; #points to indices within the lists x and y
    
    #terminate when the Infinity elements are reached in both lists
    dist = 0
    while (ix < len(x)-1 or iy < len(y)-1): 
        prev_pos = pos #update the previous position
        if x[ix]<y[iy]:
            pos = x[ix];
            ix += 1; 
            cdf_x += 1.0/(len(x)-1);
        else:
            pos = y[iy];
            iy += 1; 
            cdf_y += 1.0/(len(y)-1);
    
        dist = max(dist, abs(cdf_x - cdf_y))
    return dist**2