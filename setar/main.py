import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# rpy2 imports
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r

if __name__ == "__main__":
    name = "Q14640"
    df = pd.read_csv(f"c:\git\segmentation\data\M4\{name}.csv",usecols=[1])
    y  = pd.Series(df.iloc[:,0])
    
    # ---------------------------
    # 2️⃣ R setup
    # ---------------------------
    tsDyn = importr("tsDyn")
    base = importr("base")
    
    r_y = ro.FloatVector(y.values)
    
    # ---------------------------
    # 3️⃣ Fit SETAR model
    # ---------------------------
    model = tsDyn.setar(r_y,
                        m=2,
                        thDelay=1,
                        nthresh=1,
                        trim=0.3)
    
    print(base.summary(model))
    
    # ---------------------------
    # 4️⃣ Extract fitted values
    # ---------------------------
    fitted_r = np.array(model.rx2("fitted.values"))
    
    # ---------------------------
    # 5️⃣ Extract threshold safely
    # ---------------------------
    th_obj = model.rx2("th")
    if th_obj != ro.r('NULL') and len(th_obj) > 0:
        threshold = float(th_obj[0]) if hasattr(th_obj, "__len__") else float(th_obj)
        print("Estimated threshold:", threshold)
    else:
        threshold = None
        print("No threshold found, skipping regime coloring.")
    
    # ---------------------------
    # 6️⃣ Extract thDelay safely
    # ---------------------------
    th_delay_obj = model.rx2("thDelay")
    if th_delay_obj != ro.r('NULL'):
        th_delay = int(th_delay_obj[0]) if hasattr(th_delay_obj, "__len__") else int(th_delay_obj)
        print("Threshold delay:", th_delay)
    else:
        th_delay = 1
        print("thDelay not found, using default =", th_delay)
    
    # ---------------------------
    # 7️⃣ Plot original series and regimes (robust)
    # ---------------------------
    
    # Safe extraction of AR order
    m_obj = model.rx2("m")
    if m_obj != ro.r('NULL'):
        m_order = int(m_obj[0]) if hasattr(m_obj, "__len__") else int(m_obj)
    else:
        m_order = 2  # fallback to the AR order you supplied
    print("Using AR order:", m_order)
    
    # Align fitted values
    fitted_aligned = pd.Series(fitted_r, index=y.index[m_order:])  # skip first m_order points
    
    # Lagged series for threshold
    y_lagged = y.shift(th_delay)
    
    # Only consider indices where fitted values exist
    valid_idx = y.index[m_order:]
    
    if threshold is not None:
        regime1 = y_lagged[valid_idx] <= threshold
        regime2 = y_lagged[valid_idx] > threshold
        
        plt.figure(figsize=(12, 5))
        plt.plot(y, color='black', label='Original series', zorder=1)
        
        # Shade regimes
        plt.fill_between(valid_idx, y.min() - 1, y.max() + 1, where=regime1,
                         facecolor='lightblue', alpha=0.2, label='Regime 1 region', zorder=0)
        plt.fill_between(valid_idx, y.min() - 1, y.max() + 1, where=regime2,
                         facecolor='salmon', alpha=0.2, label='Regime 2 region', zorder=0)
        
        # Plot fitted values per regime
        plt.plot(fitted_aligned.index[regime1], fitted_aligned[regime1], '--', color='blue', label='Fitted Regime 1')
        plt.plot(fitted_aligned.index[regime2], fitted_aligned[regime2], '--', color='red', label='Fitted Regime 2')
        
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"SETAR Model Fit (Threshold={threshold:.2f})")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    else:
        # No threshold found — just plot original series
        plt.figure(figsize=(12, 5))
        plt.plot(y, color='black', label='Original series')
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("SETAR Model Fit (No threshold found)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    print("fine")