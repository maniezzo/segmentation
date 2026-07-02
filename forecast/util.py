import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import ast   # abstract syntax tree, generate python code
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import linregress


# legge dati di input
def read_series(name):
   df = pd.read_csv("C:\git\segmentation\data\M3\M3month.csv")
   idSeries = np.where(df['Series'] == name)[0][0]
   ds = pd.to_numeric(df.iloc[idSeries,6:].dropna(), errors="coerce")
   return df.iloc[idSeries,0],ds

# predizioni arima dato il modello
def reconstruct_arima(t1, t2, model, y):
   y_train = y[t1:t2]
   # Fit ARIMA
   model = ARIMA(y_train, order=model)
   fitted = model.fit()
   
   # predicted values (IN-SAMPLE)
   pred = fitted.predict()
   return pred


# home-made reconstruction, it might be wrong
def reconstruct_theta(t1, t2, y, m):
   y_train = np.array(y[t1:t2], dtype=float)
   n = len(y_train)
   
   theta_model = ThetaModel(y_train, period=m)
   fitted = theta_model.fit()
   
   alpha = fitted.params['alpha']
   b0 = fitted.params['b0']
   
   # Replicate multiplicative deseasonalization
   cma = np.convolve(y_train, np.ones(m) / m, mode='valid')
   offset = m // 2
   ratios = np.array([y_train[i + offset] / cma[i] for i in range(len(cma))])
   
   seas_factors = np.ones(m)
   for s in range(m):
      idx = [i for i in range(len(ratios)) if (i + offset) % m == s % m]
      if idx:
         seas_factors[s] = np.mean([ratios[i] for i in idx])
   seas_factors /= seas_factors.mean()
   
   sf_full = np.array([seas_factors[t % m] for t in range(n)])
   deseas = y_train / sf_full
   
   # Recover b1 via OLS on deseasonalized series (as statsmodels does internally)
   t_idx = np.arange(n)
   b1, _, _, _, _ = linregress(t_idx, deseas)
   # b0 from params is the statsmodels intercept; use it directly
   trend = b0 + b1 * t_idx
   
   # SES on deseasonalized series (one-step-ahead)
   L = np.empty(n)
   L[0] = deseas[0]
   for t in range(1, n):
      L[t] = alpha * deseas[t - 1] + (1 - alpha) * L[t - 1]
   
   # Theta combination and reseasonalize
   combined = 0.5 * (L + trend)
   in_sample_preds = combined * sf_full
   
   return in_sample_preds


# another home-made reconstruction for theta
def reconstruct_theta2(t1, t2, y, m):
   burnin = m
   if (m < 6): burnin = 6  # proprio il minimo numero di osservazioni
   preds = [None] * burnin
   period = max(1, m)
   
   for i in range(t1 + burnin, t2):
      y_train = y[t1:i]
      fitted = ThetaModel(y_train, period=period).fit()
      pred = fitted.forecast(steps=1).iloc[0]
      preds.append(pred)
   
   return preds

# ricostruzione ETS Holt-Winters
def reconstruct_HW():
   return


# plot della soluzione: nome serie, lista var in sol, df segmenti, df punti serie
def plotSol(name, lstVar, dfdata, dfpoints, yBase, yfore, model):
   ymin = dfpoints.min()
   ymax = dfpoints.max()
   yrange = ymax - ymin
   y = dfpoints.values.ravel()
   #y = dfpoints
   m = 0
   
   fig, ax = plt.subplots(figsize=(10, 6))
   ax.plot(dfpoints, marker='.', linewidth=0, color='blue')
   ax.plot(range(len(dfpoints)-6,len(dfpoints)),yBase,color="green")
   ax.plot(range(len(dfpoints)-6,len(dfpoints)),yfore,color="red")
   for i in lstVar:
      t1, t2, dfmodel = dfdata.iloc[i, 0], dfdata.iloc[i, 1], dfdata.iloc[i, 4]
      x = range(t1, t2)
      if (model == "theta"):
         ypred = reconstruct_theta2(t1, t2, y, m)
      elif (model == "HW"):
         ypred = reconstruct_HW(t1, t2, y, m)
      elif (model == "ARIMA"):
         model = ast.literal_eval(dfmodel)
         ypred = reconstruct_arima(t1, t2, model, y)
      else:
         print("No acceptable model")
         return
      print(t1, t2, x)
      ax.plot(x, ypred, linewidth=3, color="red", label=f'm={m}')
      ax.vlines(x=t2, ymin=0, ymax=ymax + 0.5 * yrange, ls='dashed', color="lightgrey")
   # plt.legend()
   plt.ylim(ymin - 0.5 * yrange, ymax + 0.5 * yrange)
   plt.title(name)
   plt.savefig(f"data/{name}.eps", format="eps")
   plt.show()
   return
