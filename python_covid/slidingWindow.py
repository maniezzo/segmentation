import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
'==================================================================
'"Segmenting Time Series: A Survey and Novel Approach"
'Eamonn Keogh, Selina Chu, David Hart, Michael Pazzani
'For simplicity, all input series here are assumed to have regular intervals
'If they are not or if there are missing data, one will need to impute the gaps
'==================================================================
'''

def calc_seg_err(x, errType = "SSE", fitType = "INTERPOL"):
    n = len(x)

    #Two data points only, trivial solution
    if (n <= 2):
        calc_seg_err = 0
        return calc_seg_err

    x_mean = 0
    x_min = float('inf')
    x_max = float('-inf')
    for i in range(1,n):
        x_mean = x_mean + x[i]
        if (x[i] > x_max): x_max = x[i]
        if (x[i] < x_min): x_min = x[i]

    x_mean = x_mean / n

    #Flat line, trivial solution
    if (x_max == x_min):
        calc_seg_err = 0
        return calc_seg_err

    if fitType == "INTERPOL":
        x_err = 0
        x_slope = (x(n) - x(1)) / (n - 1)
        for i in range(1,n):
            tmp_x = (i - 1) * x_slope + x(1)
            x_err = x_err + (x(i) - tmp_x) ^ 2

    elif fitType == "REGRESSION":
        x_slope = 0
        i_mean = (n + 1) / 2
        #tmp_z = n * (n * n - 1) / 12
        for i in range(1,n):
            x_slope = x_slope + (i - i_mean) * (x[i] - x_mean)
        x_slope = (x_slope * 12 / n) / (n * n - 1)
        x_intercept = x_mean - x_slope * i_mean

        x_err = 0
        for i in range(1,n):
            tmp_x = i * x_slope + x_intercept
            x_err = x_err + (x[i] - tmp_x)*(x[i] - tmp_x)

    else:
        print(f"calc_seg_err: {fitType} is not supported.")
        return -1

    if errType == "SSE":
        calc_seg_err = x_err
    elif errType == "SSE_NORM":
        calc_seg_err = math.sqrt(x_err) / (x_max - x_min)
    else:
        print(f"calc_seg_err: {errType} is not supported.")
        return -1
    return calc_seg_err

'''
'===========================================
'Sliding window approach to linear segmentation
'can be used in online data
'============================================
'Input: x(1:N), time series of length N
'       x_threshold, segmentation error must be lower than this threshold
'       step_size, steps to wait until a new segment is evaluated
'       errType, cost function of segmentation, SSE is sum of square, SSE_NORM is sqr(SSE)/(max-min)
'       fitType, "INTERPOL": directly joining two points, "REGRESSION": least square fit between two data points
'Output: i_anchor(1:m+1), m starting points of each segment, the m+1-th entry is simply N for convenience
'''

def Seg_Sliding(i_anchor, x, x_threshold = 1, step_size = 1, errType  = "SSE_NORM", fitType  = "INTERPOL"):
    n = len(x)
    x_tmp = []
    
    i_anchor
    n_anchor = 1
    i_anchor[1] = 1
    prev_anchor = 1
    
    x_tmp = np.zeros(4, dtype=int).reshape(2,2)
    x_tmp[1] = x[1]
    k = 1
    
    for i in range(2,n,step_size):
        #ReDim Preserve x_tmp(1 To k + step_size)
        nelem = k + step_size - len(x_tmp)
        x_tmp = x_tmp + np.zeros(nelem, dtype=int).tolist()
        for j in range(1,step_size):
            x_tmp[k + j] = x[prev_anchor + k + j - 1]

        k = k + step_size
        
        if (calc_seg_err(x_tmp, errType, fitType) >= x_threshold):
            n_anchor = n_anchor + 1
            #ReDim Preserve i_anchor(1 To n_anchor)
            nelem = n_anchor - len(i_anchor)
            i_anchor = i_anchor + np.zeros(nelem, dtype=int).tolist()

            i_anchor[n_anchor] = i - 1
            prev_anchor = i - 1
            
            #ReDim x_tmp(1 To 1)  ????
            x_tmp[1] = x[i-1]
            k = 1
    
    #ReDim Preserve i_anchor(1 To n_anchor + 1)
    nelem = n_anchor + 1 - len(i_anchor)
    i_anchor = i_anchor + np.zeros(nelem, dtype=int).tolist()
    i_anchor[n_anchor + 1] = n
    
'''
'===========================================
'Bottom-up approach to linear segmentation
'============================================
'Input: x(1:N), time series of length N
'       x_threshold, segmentation error must be lower than this threshold, defined as % of maximum segmentation error if joining start and end points directly
'       n_segment, target number of segments, override x_threshold if provided.
'       errType, cost function of segmentation, SSE is sum of square, SSE_NORM is sqr(SSE)/(max-min)
'       fitType, "INTERPOL": directly joining two points, "REGRESSION": least square fit between two data points
'       sign_penalty, penalize a merge if slopes of the two segments do not have the same signs
'Output: i_anchor(1:m+1), m starting points of each segment, the m+1-th entry is simply N for convenience
'''
def Seg_BottomUp(i_anchor, x, x_threshold = 0.05, n_segment = -1, errType = "SSE", fitType = "INTERPOL", sign_penalty = 0):
    n = len(x)
    x_tmp = []
    y_tmp = []
    #Initilize all consecutive points as segments
    n_anchor    = n - 1
    x_err_total = 0
    i_anchor = np.zeros(n_anchor+1, dtype=int)
    x_err    = np.zeros(n_anchor+1, dtype=int)
    for i in range(1,n_anchor):
        i_anchor[i] = i
    
    #Error if joining start and end by a straight line
    x_err_max = calc_seg_err(x, errType, fitType)

    #Calculate merge cost of each segment with the one after it
    x_cost_min = float('inf')
    x_cost = np.zeros(n_anchor+1, dtype=int)
    for i in (1,n_anchor - 1):

        if (i == n_anchor - 1):
            m = int(n - i_anchor[i] + 1)
        else:
            m = int(i_anchor[i + 2] - i_anchor[i] + 1)

        #ReDim x_tmp(1 To m)
        nelem = int(m - len(x_tmp))
        x_tmp = x_tmp + np.zeros(nelem, dtype=int).tolist()
        for j in range(1,m):
            x_tmp[j] = x[i_anchor[i] + j - 1]

        x_cost[i] = calc_seg_err(x_tmp, errType, fitType) - x_err[i] - x_err[i+1]
        
        if (x_cost[i] < x_cost_min):
            x_cost_min = x_cost[i]
            i_min = i
    
    if (n_segment > 0):
        isStop = n_anchor <= n_segment
    else:
        isStop = x_err_total > (x_threshold * x_err_max)
    
    #Merge segment with lowest merge cost until stopping criteriea is met
    while (isStop == False):
        #Merge i_min and i_min+1
        x_err_total = x_err_total - x_err[i_min] - x_err[i_min + 1]
        for i in range (i_min + 1,n_anchor - 1):
            i_anchor[i] = i_anchor[i + 1]
            x_err[i] = x_err[i + 1]

        for i in range(i_min + 1,n_anchor - 2):
            x_cost[i] = x_cost[i + 1]

        n_anchor = n_anchor - 1
        if (n_anchor == 1):
			#Then Exit Do
			#ReDim Preserve i_anchor(1 To n_anchor)
         nelem = n_anchor - len(i_anchor)
         i_anchor = i_anchor + np.zeros(nelem, dtype=int).tolist()
			#ReDim Preserve x_err(1 To n_anchor)
         nelem = n_anchor - len(x_err)
         x_err = x_err + np.zeros(nelem, dtype=int).tolist()
         #ReDim Preserve x_cost(1 To n_anchor - 1)
         nelem = n_anchor - 1 - len(x_cost)
         x_cost = x_cost + np.zeros(nelem, dtype=int).tolist()
         return
        
        #Calculate new error in merged segment
        if (i_min == n_anchor):
            m = n - i_anchor[i_min] + 1
        else:
            m = i_anchor[i_min + 1] - i_anchor[i_min] + 1

        #ReDim x_tmp(1 To m)
        nelem = m - len(x_tmp)
        x_tmp = x_tmp + np.zeros(nelem, dtype=int).tolist()
        for j in range(1,m):
            x_tmp[j] = x[i_anchor[i_min] + j - 1]

        x_err[i_min] = calc_seg_err(x_tmp, errType, fitType)
        x_err_total = x_err_total + x_err[i_min]
        
        #Update merge cost of i_min-1 and i_min
        if(i_min == 1): valmin = 1
        else: valmin = i_min - 1
        if(i_min == n_anchor): valmax = n_anchor - 1
        else: valmax = i_min
        for i in range(valmin, valmax):
            if (i == n_anchor - 1):
                k = n
            else:
                k = i_anchor[i + 2]
            
            #extra penalty if signs of two slopes are different
            tmp_y = (x[k] - x[i_anchor[i + 1]]) / (k - i_anchor[i + 1])
            tmp_x = (x[i_anchor[i + 1]] - x[i_anchor[i]]) / (i_anchor[i + 1] - i_anchor[i])
            if (tmp_y*tmp_x >= 0): # stesso segno
                tmp_z = 0
            else:
                tmp_z = sign_penalty * x_err_total #* Abs(tmp_y - tmp_x) / (Abs(tmp_x) + Abs(tmp_y))

            m = k - i_anchor[i] + 1
            #ReDim x_tmp(1 To m)
            nelem = m - len(x_tmp)
            x_tmp = x_tmp + np.zeros(nelem, dtype=int).tolist()
            for j in range(1,m):
                x_tmp[j] = x[i_anchor[i] + j - 1]

            x_cost[i] = calc_seg_err(x_tmp, errType, fitType) - x_err[i] - x_err[i+1] + tmp_z
        
        #New segments with mininum merge cost
        x_cost_min = float('inf')
        for i in range(1,n_anchor - 1):
            if (x_cost[i] < x_cost_min):
                x_cost_min = x_cost[i]
                i_min = i
        
        if (n_segment > 0):
            isStop = n_anchor <= n_segment
        else:
            isStop = x_err_total > (x_threshold * x_err_max)
    # end while
    
    #ReDim Preserve i_anchor(1 To n_anchor + 1)
    nelem = n_anchor + 1 - len(i_anchor)
    i_anchor = i_anchor + np.zeros(nelem, dtype=int).tolist()
    i_anchor[n_anchor + 1] = n

if __name__ == "__main__":
    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    n_anchor = 0
    n_segment = 0
    i_anchor = []
    sign_penalty = 0

    y = [1,2,3,4,5,8,10,12,14,16,18,20,21,22,23,24]
    n = len(y)
    x = np.zeros(n, dtype=int)
    x_date = np.zeros(n, dtype=int)
    for i in range(1,n):
        x_date[i] = i
        x[i] = y[i] # must rename

    sign_penalty = 0
    n_segment    = 50 # must be told explicitly
    Seg_BottomUp(i_anchor, x, 0.05, n_segment, "SSE", "REGRESSION", sign_penalty)
    m = len(i_anchor)
    v_out = np.zeros(m*2, dtype=int).reshape(m,2)
    for i in range(1,m):
        v_out[i, 1] = x_date[i_anchor[i]]
        v_out[i, 2] = x[i_anchor[i]]

    print(v_out)
