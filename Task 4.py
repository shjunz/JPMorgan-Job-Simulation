#!/usr/bin/env python
# coding: utf-8

# # Bucket FICO scores

# In[1]:


import numpy as np
import pandas as pd


# In[20]:


loan_data = pd.read_csv("Task 3 and 4_Loan_Data.csv")

fico_score = np.array(loan_data['fico_score'])
default = np.array(loan_data['default'])


# # MSE

# In[22]:


# Mean Square Error
def quantize_mse(fico_score, K, max_iters=100, seed=42):
    np.random.seed(seed)
    centroids = np.random.choice(fico_score, K, replace=False).astype(float)

    for _ in range(max_iters):
        distances = np.abs(fico_score[:, None] - centroids[None, :])
        closest_centroids = np.argmin(distances, axis=1)

        new_centroids = np.array([
            fico_score[closest_centroids == k].mean() 
            if np.any(closest_centroids == k) else centroids[k]
            for k in range(K)
        ])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    centroids = np.sort(centroids)

    # Boundaries are midpoints between centroids
    centroids = np.sort(centroids)
    boundaries = (centroids[:-1] + centroids[1:]) / 2

    fico_min, fico_max = 300, 850
    full_bounds = np.concatenate([[fico_min], boundaries, [fico_max]])
    full_bounds = np.round(full_bounds).astype(int)

    return full_bounds

# Test
full_bounds = quantize_mse(fico_score, K=5)

for i in range(len(full_bounds) - 1):
    low, high = full_bounds[i], full_bounds[i+1]
    rating = len(full_bounds) - 1 - i
    print(f"Rating {rating}: FICO {low} - {high}")


# # Dynamic Programming

# In[29]:


def quantize_ll(fico_score, default, K):
    '''
    Dynamic programming
    We do not need to compute the likelihood every time from scratch.
    Instead, we can precompute all log-likelihood for first split and then use it for the next splits.
    '''
    # Sort all FICO scores and corresponding defaults
    ind = np.argsort(fico_score)
    fico_sorted = fico_score[ind]
    default_sorted = default[ind]

    # Record all FICO scores and counts
    unique_fico, inverse, n_counts = np.unique(fico_sorted, 
                                               return_inverse=True, 
                                               return_counts=True)

    # Number of defaults for each FICO score
    k_counts = np.bincount(inverse, weights=default_sorted)

    n_total = np.concatenate([[0], np.cumsum(n_counts)])
    k_total = np.concatenate([[0], np.cumsum(k_counts)])

    # Log likelihood between boundary i and j
    def log_likelihood(i, j):
        n = n_total[j] - n_total[i]
        k = k_total[j] - k_total[i]
        if n == 0:
            return -np.inf
        p = k / n
        ll = 0.0
        if k > 0:
            ll += k * np.log(p)
        if k < n:
            ll += (n - k) * np.log(1 - p)
        return ll

    # Dynamic programming
    N = len(unique_fico)
    dp = np.full((K + 1, N + 1), -np.inf)
    parent = np.zeros((K + 1, N + 1), dtype=int)
    dp[0][0] = 0.0

    for b in range(1, K + 1):
        for j in range(b, N + 1):
            for i in range(b - 1, j):
                cand = dp[b-1][i] + log_likelihood(i, j)
                if cand > dp[b][j]:
                    dp[b][j] = cand
                    parent[b][j] = i

    # Backtrack to find splits
    splits = []
    j = N
    for b in range(K, 0, -1):
        i = parent[b][j]
        if i > 0:
            splits.append(i)
        j = i
    splits.reverse()

    # Take the mid point as boundaries
    boundaries = [(unique_fico[s-1] + unique_fico[s]) / 2 for s in splits]
    full_bounds = np.concatenate([[300], boundaries, [850]])
    return np.round(full_bounds).astype(int)

# Test
bounds_ll = quantize_ll(fico_score, default, K=5)

for i in range(len(bounds_ll) - 1):
    rating = len(bounds_ll) - 1 - i
    print(f"Rating {rating}: FICO {bounds_ll[i]} - {bounds_ll[i+1]}")

