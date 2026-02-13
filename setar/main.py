import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri

# Activate conversion
converter = default_converter + pandas2ri.converter

# Example series
y = pd.Series(np.random.normal(size=300))

# Convert pandas -> R
with localconverter(converter):
    r_y = ro.conversion.py2rpy(y)

# Load tsDyn
tsDyn = importr("tsDyn")
base = importr("base")

# Fit SETAR
model = tsDyn.setar(r_y, m=2, thDelay=1, nthresh=1)

print(base.summary(model))
