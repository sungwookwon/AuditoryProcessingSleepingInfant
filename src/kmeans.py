## -------- Functions used to build K-Means clustering algorithm ------ ##

from tqdm import tqdm 
import numpy as np

# --------- Basic functions ------------ #

def compute_distances(X, centers):
    """
    Compute squared Euclidean distances between each row of X and each center using broadcasting.
    X : (n, d) numpy array
    centers : (K, d) numpy array
    returns d2 : (n, K) numpy array of distances between x_i and center_j
    """
    # Use broadcasting to compute all distances at once
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]  # Shape: (n, K, d)
    return np.sum(diff**2, axis=2)  # Shape: (n, K)

def assign_labels(X, centers):
    '''
    X : (n, d) numpy array
    centers : (K, d) numpy array
    returns assignments : (n,) numpy array of cluster indices
    '''
    distances = compute_distances(X, centers)       # Compute distances
    assignments = np.argmin(distances, axis=1)      # Assign to closest center
    return assignments

def compute_centers(X, labels, K):
    '''
    X : (n, d) numpy array
    labels : (n,) numpy array of cluster indices
    K : number of clusters
    returns centers : (K, d) numpy array of new center positions
    '''
    n, d = X.shape
    centers = np.zeros((K, d), dtype=float)
    for k in range(K):
        members = X[labels == k]            # Extract the submatrix of X that have labels k. It is a matrix of size (|C_k|, d)
        centers[k] = members.mean(axis=0)   # Compute the mean of each column of this submatrix. It is a vector of size d
    return centers

def compute_cost(X, centers, labels):
    '''
    X : (n, d) numpy array
    centers : (K, d) numpy array
    labels : (n,) numpy array of cluster indices
    returns cost : float, the K-Means cost function evaluated on X with given centers and labels
    '''
    centers_per_datapoint = centers[labels]         # Shape (n, d): for each point, get the center it is assigned to
    cost = np.sum((X - centers_per_datapoint)**2)   # Compute the sum of squared distances between each point and its assigned center
    return cost


# -------------- Run the algorithm --------------- #

### k-means

def kmeans_numpy(X, K, N_iters=100, seed=None):
    """
    X : (n, d) numpy array of data points
    K : number of clusters
    N_iters : maximum number of iterations
    seed : random seed for reproducibility
    returns centers_history : (N_iters+1, K, d) numpy array of centers over iterations
            labels : (n,) numpy array of final cluster assignments
            cost_history : list of costs over iterations
            final_cost : final cost value
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # --- Initialization (random training points) ---
    initial_idx = rng.choice(n, size=K, replace=False)
    centers = X[initial_idx].copy()

    # --- Storing vectors ---
    centers_history = [centers.copy()]
    labels = np.zeros(n, dtype=int)
    cost_history = []

    # --- Training loop ---
    for it in tqdm(range(N_iters)):       # Loop for a maximum of N_iters iterations
    
        # 1) Compute the assignments and cost
        assignments = assign_labels(X, centers)             # Compute the assignments for each datapoint
        cost = compute_cost(X, centers, assignments)        # Compute the cost for that assignment

        # 2) Compute the new centers
        new_centers = compute_centers(X, assignments, K)    # Compute new center positions
        
        # 3) Update for the next iteration
        centers = new_centers                                # Update centers
        labels = assignments                                 # Update labels
        
        # Store history
        cost_history.append(cost)                            # Store the cost
        centers_history.append(new_centers.copy())           # Store the center positions

    # Final cost computation
    final_cost = compute_cost(X, centers, labels)
    cost_history.append(final_cost)
    
    return np.array(centers_history), labels, cost_history
