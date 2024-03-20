"""
This file contains the implementation of the DTW distance and lower bounding measures.

Author:
- Artur Dandolini Pescador
"""

import numpy as np

def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dp = np.zeros((n+1, m+1))
    for i in range(1, n+1):
        dp[i][0] = np.inf
    for i in range(1, m+1):
        dp[0][i] = np.inf
    dp[0][0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = (s1[i-1] - s2[j-1]) ** 2
            dp[i][j] = cost + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return np.sqrt(dp[n][m])

def lb_keogh(s1, s2, r):
    """
    Calculate the LB_Keogh lower bound between two time series.
    """
    LB_sum = 0
    for ind, i in enumerate(s1):

        lower_bound = np.min(s2[(max(0, ind - r)):(min(len(s2), ind + r + 1))])
        upper_bound = np.max(s2[(max(0, ind - r)):(min(len(s2), ind + r + 1))])

        if i > upper_bound:
            LB_sum = LB_sum + (i - upper_bound)**2
        elif i < lower_bound:
            LB_sum = LB_sum + (i - lower_bound)**2

    return np.sqrt(LB_sum)

def paa(data, bins):
    """
    Compute the Piecewise Aggregate Approximation (PAA) of a time series
    """
    # Calculate the length of the time series and the size of each segment
    length = len(data)
    step = length // bins
    
    # Calculate the mean of each segment
    means = [np.mean(data[i:i+step]) for i in range(0, length, step)]
    
    # Repeat the means for each segment
    paa = np.repeat(means, step)
    
    return paa

def lb_paa(Q, C_bar, U_hat, L_hat):
    """
    Calculate the LB_PAA lower bounding measure between a query and a candidate time series in PAA representation.
    """
    LB_sum = 0
    for i in range(len(C_bar)):
        if C_bar[i] > U_hat[i]:
            LB_sum += (C_bar[i] - U_hat[i])**2
        elif C_bar[i] < L_hat[i]:
            LB_sum += (C_bar[i] - L_hat[i])**2
    return np.sqrt(LB_sum)

def create_paa_bounds(x, r, dim):
    """
    Create the Piecewise Aggregate Approximation bounds for a time series.
    """
    U = [max(x[max(i-r, 0):min(i+r+1, len(x))]) for i in range(len(x))]
    L = [min(x[max(i-r, 0):min(i+r+1, len(x))]) for i in range(len(x))]
    U_hat = paa(U, dim)
    L_hat = paa(L, dim)
    return U_hat, L_hat

def mindist(Q, MBR):
    """
    Calculate the minimum distance between a query and a Minimum Bounding Rectangle (MBR).
    """
    mindist = 0
    for i in range(len(Q)):
        if Q[i] > MBR[i][1]:
            mindist += (Q[i] - MBR[i][1])**2
        elif Q[i] < MBR[i][0]:
            mindist += (MBR[i][0] - Q[i])**2
    return np.sqrt(mindist)