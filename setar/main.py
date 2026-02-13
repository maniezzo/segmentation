import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# rpy2 imports
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r

if __name__ == "__main__":
    name = "N1881"
    df = pd.read_csv(f"c:\git\segmentation\data\M3\{name}.csv",usecols=[1])
    y  = pd.Series(df.iloc[:,0])
    
    # R setup
    tsDyn    = importr("tsDyn")
    base     = importr("base")
    graphics = importr("graphics")  # optional if you want R plots
    
    # Convert to R vector
    r_y = ro.FloatVector(y.values)
    
    # Fit SETAR model
    model = tsDyn.setar(r_y,
                        m=2,        # AR order
                        thDelay=1,  # threshold delay, how many instants ago I react to
                        nthresh=1,  # number of thresholds, num  of regimes
                        trim   = 0.3) # min percentage of data in each regime
    
    # Print R summary
    print(base.summary(model))
    
    # Extract fitted values
    fitted_r = np.array(model.rx2("fitted.values"))
    
    # Plot original data and fitted values
    plt.figure(figsize=(12,5))
    plt.plot(y, label="Original series", color="blue")
    plt.plot(fitted_r, label="SETAR fitted values", color="red", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"SETAR series {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("fine")