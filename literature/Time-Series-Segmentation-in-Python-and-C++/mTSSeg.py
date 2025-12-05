
'''
==================================================================
"Segmenting Time Series: A Survey and Novel Approach"
Eamonn Keogh, Selina Chu, David Hart, Michael Pazzani
For simplicity, all input series here are assumed to have regular intervals
If they are not or if there are missing data, one will need to impute the gaps
==================================================================

===========================================
Sliding window approach to linear segmentation
can be used in online data
============================================
Input: x(1:N), time series of length N
       x_threshold, segmentation error must be lower than this threshold
       step_size, steps to wait until a new segment is evaluated
       errType, cost function of segmentation, SSE is sum of square, SSE_NORM is sqr(SSE)/(max-min)
       fitType, "INTERPOL": directly joining two points, "REGRESSION": least square fit between two data points
Output: i_anchor(1:m+1), m starting points of each segment, the m+1-th entry is simply N for convenience
'''
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from typing import List

def calc_seg_err(x, errType = "SSE", fitType = "INTERPOL"):
   '''
    Dim i As Long, j As Long, k As Long, m As Long, n As Long
    Dim tmp_x As Double, tmp_y As Double, tmp_z As Double
    Dim x_err As Double
    Dim x_mean As Double, x_max As Double, x_min As Double, i_mean As Double
    Dim x_slope As Double, x_intercept As Double, t_mean As Double)
   '''
   n = len(x)
    
   #Two data points only, trivial solution
   if n <= 2:
      calc_seg_err = 0
      return calc_seg_err
    
   x_mean = np.mean(x)
   x_min  = np.min(x)
   x_max  = np.max(x)

   # Flat line, trivial solution
   if (x_max == x_min):
      calc_seg_err = 0
      return calc_seg_err

   if fitType == "INTERPOL":
      x_err = 0
      x_slope = (x[-1] - x[0]) / (n - 1)
      for i in range(0,n):
         tmp_x = (i - 1) * x_slope + x[0]
         x_err = x_err + (x[i] - tmp_x)**2
   elif fitType == "REGRESSION":
      x_slope = 0
      i_mean = (n + 1) / 2
      #tmp_z = n * (n * n - 1) / 12
      for i in range(0,n):
         x_slope = x_slope + (i - i_mean) * (x[i] - x_mean)
      x_slope = (x_slope * 12 / n) / (n * n - 1)
      x_intercept = x_mean - x_slope * i_mean
        
      x_err = 0
      for i in range(0,n):
         tmp_x = i * x_slope + x_intercept
         x_err = x_err + (x[i] - tmp_x)**2
   else:
      print(f"calc_seg_err: {fitType} is not supported.")

   if errType == "SSE":
      calc_seg_err = x_err
   elif errType == "SSE_NORM":
      calc_seg_err = np.sqrt(x_err) / (x_max - x_min)
   else:
      print(f"calc_seg_err: {errType} is not supported.")
   return calc_seg_err

'''
===========================================
Sliding window approach to linear segmentation.
This function implements a greedy sliding window approach to linear segmentation. 
It starts a segment at the first point, then extends the segment by step_size points 
at a time until the error exceeds a given threshold (x_threshold). 
When the error limit is reached, the anchor for the new segment is set at the point 
before the failing point.
============================================
Input: x(1:N), time series of length N
       x_threshold, segmentation error must be lower than this threshold
       step_size, steps to wait until a new segment is evaluated
       errType, cost function of segmentation, SSE is sum of square, SSE_NORM is sqr(SSE)/(max-min)
       fitType, "INTERPOL": directly joining two points, "REGRESSION": least square fit between two data points
Output: i_anchor(1:m+1), m starting points of each segment, the m+1-th entry is simply N for convenience
'''
def seg_sliding(x: np.ndarray, x_threshold: float = 1.0, step_size: int = 1,
                err_type: str = "SSE_NORM", fit_type: str = "INTERPOL") -> List[int]:
   """
   Performs sliding window linear segmentation on a time series.

   Input:
       x (np.ndarray): Time series data (0-based indexing).
       x_threshold (float): Segmentation error threshold.
       step_size (int): Number of points to extend the segment by in each step.
       err_type (str): Cost function ('SSE_NORM', 'SSE', etc.).
       fit_type (str): Fitting method ('INTERPOL' or 'REGRESSION').

   Output:
       List[int]: Anchor indices (start points of each segment, 0-based).
                  The last element is the total length N for convenience.
   """
   N = len(x)
   if N <= 1:
      return [0, N]

   # --- Initialization ---
   # i_anchor will hold the 0-based index of the starting point of each segment
   i_anchor = [0]
   prev_anchor = 0

   # x_tmp holds the data points of the current segment being tested.
   current_segment_data = [x[0]]

   # The loop iterates through indices i = 1 to N-1 (Python range(1, N))
   step_size = max(1, step_size)
   i = 1  # Start checking from the second point

   # --- Sliding Window Loop ---
   while i < N:
      # Determine the range of points to add in this step, covers points from i up to i + step_size - 1.
      # The segment being tested goes from prev_anchor up to the end of this range.
      end_idx = min(N, i + step_size)

      # Get the full data of the segment being tested:
      segment_to_test = x[prev_anchor: end_idx]

      # Check if the error exceeds the threshold
      if calc_seg_err(segment_to_test, err_type, fit_type) >= x_threshold:
         #  Threshold exceeded: Finalize current segment and start a new one ---
         new_anchor = i - 1
         i_anchor.append(new_anchor)
         # Reset the window: The new segment starts at the new anchor (i - 1).
         prev_anchor = new_anchor
         # Reset loop index 'i' to the new anchor + 1
         i = new_anchor + 1
      else: # Threshold not exceeded: Continue extending the window
         i = end_idx

   # --- Finalization ---
   i_anchor.append(N)
   return i_anchor

# =========================================================
# Bottom-up approach to linear segmentation
# =========================================================
def seg_bottom_up(x: np.ndarray, x_threshold: float = 0.05, n_segment: int = -1,
                  err_type: str = "SSE", fit_type: str = "INTERPOL", sign_penalty: float = 0.0):
   """
   Performs bottom-up linear segmentation on a time series.

   Input:
       x (np.ndarray): Time series data (1-D array, assumes 0-based indexing in Python).
       x_threshold (float): Segmentation error threshold (as % of max error).
       n_segment (int): Target number of segments (overrides x_threshold if > 0).
       err_type (str): Cost function ('SSE' or 'SSE_NORM').
       fit_type (str): Fitting method ('INTERPOL' or 'REGRESSION').
       sign_penalty (float): Penalty coefficient for merging segments with different slope signs.

   Output:
       list[int]: Anchor indices (start points of each segment, 0-based).
                  The last element is the total length N for convenience.
   """
   N = len(x)
   if N <= 2:
      return [0, N]

   # --- Initialization ---

   # Start with every consecutive point as a segment (N-1 initial segments)
   # i_anchor holds the start index of each segment
   i_anchor = list(range(N - 1))
   n_anchor = len(i_anchor)  # Initial number of segments = N-1
   x_err_max = calc_seg_err(x, err_type, fit_type)

   # Initial error of a 2-point segment is 0 for INTERPOL/REGRESSION, as a line fits perfectly.
   x_err = [0.0] * n_anchor
   x_err_total = 0.0

   # --- Calculate initial merge costs ---
   # x_cost[i] is the cost of merging segment i (anchor i_anchor[i]) and segment i+1 (anchor i_anchor[i+1])
   x_cost = []
   x_cost_min = float('inf')
   i_min = -1

   for i in range(n_anchor - 1):
      # Segment 1: from i_anchor[i] to i_anchor[i+1]
      # Segment 2: from i_anchor[i+1] to i_anchor[i+2]

      # The merged segment runs from i_anchor[i] up to the end point of segment i+1
      # The end point of segment i+1 is the start of the next segment (i_anchor[i+2]) or N

      # Next segment's anchor (or N if it's the last one)
      anchor_k = i_anchor[i + 2] if i + 2 < n_anchor else N

      # The merged segment data
      merged_segment_data = x[i_anchor[i]: anchor_k]  # Python slice: start (inclusive) to end (exclusive)

      # Cost = Error(Merged Segment) - Error(Segment i) - Error(Segment i+1) + Penalty
      # Since initial segments are 2 points, Error(i) and Error(i+1) are 0
      cost = calc_seg_err(merged_segment_data, err_type, fit_type) - x_err[i] - x_err[i + 1]

      x_cost.append(cost)

      if cost < x_cost_min:
         x_cost_min = cost
         i_min = i

   # --- Main Loop (Merge Segments) ---
   is_stop = False
   if n_segment > 0:
      is_stop = n_anchor <= n_segment
   else:
      # Check against initial total error. Since initial x_err_total is 0, this is likely False.
      is_stop = x_err_total > (x_threshold * x_err_max)

   while not is_stop:
      if n_anchor <= 1:
         break

      # 1. Merge i_min and i_min+1
      x_err_total = x_err_total - x_err[i_min] - x_err[i_min + 1]

      # Remove segment i_min+1. This is done by shifting the subsequent anchor/error/cost lists.
      i_anchor.pop(i_min + 1)
      x_err.pop(i_min + 1)

      # Shift x_cost (i_min is removed, so we shift from i_min+1)
      if i_min < len(x_cost):  # Check if there is an element to remove
         x_cost.pop(i_min)

      n_anchor -= 1
      if n_anchor == 1:
         break

      # 2. Calculate new error in the merged segment (at index i_min)
      anchor_k = i_anchor[i_min + 1] if i_min + 1 < n_anchor else N
      merged_segment_data = x[i_anchor[i_min]: anchor_k]

      new_err = calc_seg_err(merged_segment_data, err_type, fit_type)
      x_err[i_min] = new_err
      x_err_total += new_err

      # 3. Update merge costs involving the newly merged segment (i_min)
      # The range of indices in x_cost to update is i_min-1 to i_min (if valid)
      start_idx = max(0, i_min - 1)
      end_idx = min(n_anchor - 2, i_min)  # n_anchor - 2 is the last valid index for x_cost

      for i in range(start_idx, end_idx + 1):
         # i is the index into the x_cost array.
         # Anchor k is the end point of segment i+1
         anchor_k = i_anchor[i + 2] if i + 2 < n_anchor else N

         # --- Slope Penalty Calculation
         tmp_z = 0.0
         if sign_penalty > 0 and i + 1 < n_anchor:
            # Slope of segment i+1 (from i_anchor[i+1] to anchor_k)
            if anchor_k > i_anchor[i + 1]:
               tmp_y = (x[anchor_k - 1] - x[i_anchor[i + 1]]) / (anchor_k - 1 - i_anchor[i + 1])
            else:  # Segment length 1, slope is undefined/0
               tmp_y = 0.0

            # Slope of segment i (from i_anchor[i] to i_anchor[i+1])
            if i_anchor[i + 1] > i_anchor[i]:
               tmp_x = (x[i_anchor[i + 1] - 1] - x[i_anchor[i]]) / (i_anchor[i + 1] - 1 - i_anchor[i])
            else:
               tmp_x = 0.0

            if np.sign(tmp_y) != np.sign(tmp_x):
               tmp_z = sign_penalty * x_err_total
         # --- End Slope Penalty ---
         merged_segment_data = x[i_anchor[i]: anchor_k]
         cost = calc_seg_err(merged_segment_data, err_type, fit_type) - x_err[i] - x_err[i + 1] + tmp_z

         if i < len(x_cost):
            x_cost[i] = cost
         else:
            # This should only happen if we are at the very end of the list and it needs to be appended
            # due to the shifting logic not removing the last element.
            pass

      # 4. Find new minimum merge cost
      x_cost_min = float('inf')
      i_min = -1
      for i, cost in enumerate(x_cost):
         if cost < x_cost_min:
            x_cost_min = cost
            i_min = i

      # 5. Check stopping criteria
      if n_segment > 0:
         is_stop = n_anchor <= n_segment
      else:
         is_stop = x_err_total > (x_threshold * x_err_max)

   # --- Final Output ---
   i_anchor.append(N)
   return i_anchor

if __name__ == "__main__":
   x = pd.read_csv("series.csv").values  # time series of length N
   x_threshold = 0.1    # segmentation error must be lower than this threshold
   step_size = 10       # steps to wait until a new segment is evaluated
   errType = "SSE"      # cost function of segmentation, SSE is sum of square, SSE_NORM is sqr(SSE)/(max-min)
   fitType = "INTERPOL" # "INTERPOL": directly joining two points, "REGRESSION": least square fit between two data points

   anchorsBU = seg_bottom_up(x, n_segment=12)
   print(f"Bottom up segment Anchors : {anchorsBU}")

   anchorsSW = seg_sliding(x,x_threshold = 5)
   print(f"Sliding window segment Anchors: {anchorsSW}")

   fig = plt.figure(figsize=(12,9))
   plt.plot(x,label="data")

   # Bottom up segmnets
   nSegmBU = len(anchorsBU) - 1
   for i in range(nSegmBU):
      start_index = anchorsBU[i]
      end_index = anchorsBU[i + 1] - 1
      segment_indices = np.arange(start_index, end_index + 1)
      segment_data = x[start_index: end_index + 1]
      line_x = [start_index, end_index]
      line_y = [x[start_index], x[end_index]]
      plt.plot(line_x, line_y, color='red', linewidth=2, linestyle='-', label='BottomUp segments' if i == 0 else "")
   indAnchorBU = anchorsBU[:-1]  # Exclude the final index
   anchorBU = x[indAnchorBU]
   plt.plot(indAnchorBU, anchorBU, 's', color='red', markersize=8, label='BU Anchor Points')

   # Sliding window segmnets
   nSegmSW = len(anchorsSW) - 1
   for i in range(nSegmSW):
      start_index = anchorsSW[i]
      end_index = anchorsSW[i + 1] - 1
      segment_indices = np.arange(start_index, end_index + 1)
      segment_data = x[start_index: end_index + 1]
      line_x = [start_index, end_index]
      line_y = [x[start_index], x[end_index]]
      plt.plot(line_x, line_y, color='green', linewidth=2, linestyle='-', label='SlidingWin segments' if i == 0 else "")
   indAnchorSW = anchorsSW[:-1]  # Exclude the final index
   anchorSW = x[indAnchorSW]
   plt.plot(indAnchorSW, anchorSW, 's', color='green', markersize=8, label='SW Anchor Points')

   plt.legend(loc="best")
   plt.show()

   print("fine")