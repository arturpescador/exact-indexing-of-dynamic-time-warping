"""
This file contains the implementation of the DTW distance and lower bounding measures.

Author:
- Artur Dandolini Pescador
"""

import numpy as np

def dtw_distance(s1, s2):
    """
    Compute the Dynamic Time Warping distance between two time series.

    Parameters:
    - s1: the first time series
    - s2: the second time series

    Returns:
    - The DTW distance between s1 and s2
    """
    n, m = len(s1), len(s2)
    dp = np.zeros((n+1, m+1))
    for i in range(1, n+1):
        dp[i][0] = np.inf
    for i in range(1, m+1):
        dp[0][i] = np.inf
    dp[0][0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = (s1[i-1] - s2[j-1]) ** 2 # euclidean distance
            dp[i][j] = cost + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return np.sqrt(dp[n][m])

def lb_kim(s1, s2):
    """
    Compute the LB_Kim lower bounding measure between two time series.

    Parameters:
    - s1: the first time series
    - s2: the second time series

    Returns:
    - The LB_Kim lower bound between s1 and s2
    """
    start1, end1, min1, max1 = s1[0], s1[-1], min(s1), max(s1)
    start2, end2, min2, max2 = s2[0], s2[-1], min(s2), max(s2)
    
    # Calculate lower bound
    lb = max((start1 - start2)**2, (end1 - end2)**2, (min1 - min2)**2, (max1 - max2)**2)
    return lb

def lb_yi(s1, s2):
    """
    Compute the LB_YI lower bounding measure between two time series.

    Parameters:
    - s1: the first time series
    - s2: the second time series

    Returns:
    - The LB_YI lower bound between s1 and s2
    """
    max2 = np.max(s2)
    min2 = np.min(s2)
    
    lb_sum = 0
    for point in s1:
        if point > max2:
            lb_sum += (point - max2)**2
        elif point < min2:
            lb_sum += (min2 - point)**2
    return np.sqrt(lb_sum)

def lb_keogh(s1, s2, r):
    """
    Calculate the LB_Keogh lower bound between two time series.

    Parameters:
    - s1: the first time series
    - s2: the second time series
    - r: the window size
    
    Returns:
    - The LB_Keogh lower bound between s1 and s2
    """
    s1 = np.array(s1)
    s2 = np.array(s2)
    
    LB_sum = 0
    for ind, i in enumerate(s1):
        lower_bound = np.min(s2[(max(0, ind - r)):(min(len(s2), ind + r + 1))])
        upper_bound = np.max(s2[(max(0, ind - r)):(min(len(s2), ind + r + 1))])

        if i > upper_bound:
            LB_sum = LB_sum + (i - upper_bound)**2
        elif i < lower_bound:
            LB_sum = LB_sum + (i - lower_bound)**2

    return np.sqrt(LB_sum)

def paa(timeSeries, dims): 
    """
    Compute the Piecewise Aggregate Approximation (PAA) of a time series.

    Parameters:
    - timeSeries: the time series
    - dims: the number of dimensions

    Returns:
    - The PAA representation of the time series
    """
    # Calculate the length of the time series and the size of each segment
    length = len(timeSeries)
    step = length // dims
    
    # Calculate the mean of each segment
    means = [np.mean(timeSeries[i:i+step]) for i in range(0, length, step)]
    
    # Repeat the means for each segment
    paa = np.repeat(means, step)
    
    return paa

def lb_paa(C_bar, U_hat, L_hat):
    """
    Calculate the LB_PAA lower bounding measure between a query and a candidate time series in PAA representation.

    Parameters:
    - C_bar: the candidate time series in PAA representation
    - U_hat: the upper bound in PAA representation
    - L_hat: the lower bound in PAA representation

    Returns:
    - The LB_PAA lower bound between the query and the candidate time series
    """
    LB_sum = 0
    for i in range(len(C_bar)):
        if C_bar[i] > U_hat[i]:
            LB_sum += (C_bar[i] - U_hat[i])**2
        elif C_bar[i] < L_hat[i]:
            LB_sum += (C_bar[i] - L_hat[i])**2
    return np.sqrt(LB_sum)

def create_paa_bounds(timeSeries, r, dim):
    """
    Create the Piecewise Aggregate Approximation bounds for a time series.

    Parameters:
    - timeSeries: the time series
    - r: the window size
    - dim: the number of dimensions

    Returns:
    - The upper and lower bounds in PAA representation
    """
    U = [max(timeSeries[max(i-r, 0):min(i+r+1, len(timeSeries))]) for i in range(len(timeSeries))]
    L = [min(timeSeries[max(i-r, 0):min(i+r+1, len(timeSeries))]) for i in range(len(timeSeries))]
    U_hat = paa(U, dim)
    L_hat = paa(L, dim)
    return U_hat, L_hat

def mindist(Q, MBR):
    """
    Calculate the minimum distance between a query and a Minimum Bounding Rectangle (MBR).

    Parameters:
    - Q: the query point
    - MBR: the Minimum Bounding Rectangle

    Returns:
    - The minimum distance between the query and the MBR
    """
    mindist = 0
    for i in range(len(Q)):
        if Q[i] > MBR[i][1]:
            mindist += (Q[i] - MBR[i][1])**2
        elif Q[i] < MBR[i][0]:
            mindist += (MBR[i][0] - Q[i])**2
    return np.sqrt(mindist)

