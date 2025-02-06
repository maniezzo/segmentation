import warnings
import numpy as np
import pandas as pd
import scipy
import scipy.stats as st
import statsmodels as sm
import lmfit
import matplotlib.pyplot as plt
from lmfit.models import LognormalModel

# Curve Fitting Example
def f(x, a, b, c):
	return LognormalModel(x, a, b, c)

if __name__ == "__main__":

   # Load data from statsmodels datasets
   df = pd.read_csv("../Covid_italia_22.csv")
   rollMean = df.iloc[:120,1].rolling(7).mean()
   y = rollMean[7:]
   x = np.arange(len(y))

   mod = LognormalModel()
   pars = mod.guess(y, x=x)
   out = mod.fit(y, pars, x=x)

   print(out.fit_report(min_correl=0.25))
   mod = lmfit.Model(f)
   # we set the parameters (and some initial parameter guesses)
   mod.set_param_hint("a", value=10.0, vary=True)
   mod.set_param_hint("b", value=10.0, vary=True)
   mod.set_param_hint("c", value=10.0, vary=True)
   params = mod.make_params()

   result = mod.fit(data, params, method="leastsq", x=x)  # fitting
   print(result.best_values)
   print(result)

   plt.figure(figsize=(8, 4))
   result.plot_fit(datafmt="-")
   plt.show()

   print("fine")
