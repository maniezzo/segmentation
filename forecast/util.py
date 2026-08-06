import numpy as np, pandas as pd,os
import matplotlib.pyplot as plt
import ast   # abstract syntax tree, generate python code
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import STL
from scipy.stats import linregress
from models import go_RF


# legge dati di input
def read_series(filedata):
   basedir, filename, name = filedata.split(',')
   if filename != '':
      # leggo la serie come riga da un benchmarck
      filename = os.path.join(basedir, filename)
      df = pd.read_csv(filename)
      idSeries = np.where(df['Series'] == name)[0][0]
      ds = pd.to_numeric(df.iloc[idSeries,6:].dropna(), errors="coerce") # questa è una dataseries
   else:
      # leggo la serie come colonna da un file specifico
      filename = os.path.join(basedir, f"{name}.csv")
      ds = pd.read_csv(filename, usecols=[1]).values.flatten() # questo è un array (non sembra fagli male)
   return name,ds

# predizioni arima dato il modello
def reconstruct_arima(t1, t2, model, y):
   y_train = y[t1:t2]
   # Fit ARIMA
   model = ARIMA(y_train, order=model)
   fitted = model.fit()
   
   # predicted values (IN-SAMPLE)
   pred = fitted.predict()
   return pred

# UNUSED home-made reconstruction, it might be wrong
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

# THIS ONE home-made reconstruction for theta
def reconstruct_theta2(t1, t2, y, m, coeff_seas):
   tstart = t1  # primo istante da prevedere
   idCoeff1 = tstart % m
   coeff = np.roll(coeff_seas, -idCoeff1)  # Rotate LEFT by idCoeff1 positions (negative means left)
   ysegm = y[t1:t2]
   coeff_stretch = coeff[np.arange(len(ysegm)+1) % len(coeff)]

   burnin = m
   if (m < 6): burnin = 6  # proprio il minimo numero di osservazioni
   preds = [None] * burnin
   period = max(1, m)
   
   for i in range(t1 + burnin, t2+1):  # estremi inclusi
      y_train = y[t1:i]
      fitted = ThetaModel(y_train, period=period).fit()
      pred = fitted.forecast(steps=1).iloc[0]
      preds.append(pred)
   
   preds = [p if p is None else p + c for p, c in zip(preds, coeff_stretch)]
   return preds

# ricostruzione ETS Holt-Winters
def reconstruct_HW(t1, t2, y, m, coeff_seas):
   tstart = t1  # primo istante da prevedere
   idCoeff1 = tstart % m
   coeff = np.roll(coeff_seas, -idCoeff1)  # Rotate LEFT by idCoeff1 positions (negative means left)
   ysegm = y[t1:t2]
   coeff_stretch = coeff[np.arange(len(ysegm)+1) % len(coeff)]
   
   # no seasonality (Double Exponential Smoothing)
   model = ExponentialSmoothing(y,
       trend    = "add",
       seasonal = None,
       initialization_method = "estimated")
   hwfit = model.fit()
   ypred = hwfit.predict(t1,t2)
   ypred += coeff_stretch
   
   return ypred

# ricostruzione Random forest
def reconstruct_RF(t1, t2, y, m, coeff_seas):
   tstart = t1  # primo istante da prevedere
   ysegm = y[t1:t2+1]
   
   ysub = ysegm
   rffit, _ = go_RF(ysub, 1, 6)  # keep this if you still need the recursive forecast elsewhere
   
   y = np.asarray(ysub, dtype=float).reshape(-1)
   lags = 12 if len(y) < 24 else 18
   
   X = np.array([y[i - lags:i] for i in range(lags, len(y))])
   Y = y[lags:]
   
   oob_model = RandomForestRegressor(
      n_estimators=100,
      random_state=42,
      oob_score=True,
      bootstrap=True  # required for OOB; default is already True but be explicit
   )
   oob_model.fit(X, Y)
   
   ypred = oob_model.oob_prediction_  # aligned to ysub[lags:], one prediction per row of X
   idCoeff1 = (tstart+lags) % m
   coeff = np.roll(coeff_seas, -idCoeff1)  # Rotate LEFT by idCoeff1 positions (negative means left)
   coeff_stretch = coeff[np.arange(len(ypred)) % len(coeff)]
   ypred += coeff_stretch
   ypred = np.concatenate(([None]*lags,ypred))
   return ypred

# destagionalizzazione serie
def deseason(y,m=12):
   stl = STL(y,
             period=m,
             seasonal=len(y) * 2 - 1,  # Large window forces it to look at all points
             seasonal_deg=0,  # Forces the seasonal component to be strictly periodic
             robust=True)
   res = stl.fit()
   
   coeff_seas = res.seasonal[0:12]
   trend = res.trend
   resid = res.resid

   return coeff_seas,trend,resid

# plot della soluzione: nome serie, lista var in sol, df segmenti, df punti serie
def plotSol(name, lstVar, dfModel, dfpoints, yBase, yfore, m, model):
   nforecast = len(yfore)
   ymin = dfpoints.min()
   ymax = dfpoints.max()
   yrange = ymax - ymin
   y = np.asarray(dfpoints, dtype=float)
   coeff_seas, trend, resid = deseason(y)
   y = trend+resid  # serie destagionalizzata
   m = 1
   
   fig, ax = plt.subplots(figsize=(10, 6))
   ax.plot(dfpoints, marker='.', linewidth=0, color='blue')
   ax.plot(range(len(dfpoints)-nforecast,len(dfpoints)),yBase,color="green")
   ax.plot(range(len(dfpoints)-nforecast,len(dfpoints)),yfore,color="red")
   for i in lstVar:
      t1, t2, dfmodel = dfModel.iloc[i, 0], dfModel.iloc[i, 1], dfModel.iloc[i, 4]
      x = range(t1, t2+1)  # estremi inclusi
      if (model == "theta"):
         ypred = reconstruct_theta2(t1, t2, y, m, coeff_seas)
      elif (model == "HW"):
         ypred = reconstruct_HW(t1, t2, y, m, coeff_seas)
      elif (model == "RF"):
         ypred = reconstruct_RF(t1, t2, y, m, coeff_seas)
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
   plt.title(f"{name} - {model}")
   plt.savefig(f"data/{name}{model}.eps", format="eps")
   plt.show()
   return
