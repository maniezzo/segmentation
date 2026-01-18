import numpy as np
from numpy.linalg import lstsq
from ruptures.base import BaseCost

class AICcost(BaseCost):
    """
    AIC per segment for a simple linear model:
        y_t ≈ a * t + b
        MSE = SSE / n
        AIC = n * log(MSE) + 2 * k
    """
    model = "aic_mse_linear"
    min_size = 5  # >= number of parameters + 1

    def __init__(self, k_params=2):
        self.k_params = k_params

    def fit(self, signal):
        # signal is 1D: y only; index acts as regressor
        self.signal = np.asarray(signal, dtype=float)
        return self

    def error(self, start, end):
        y = self.signal[start:end]
        n = end - start
        if n < self.min_size:
            return np.inf  # Instead of raising ValueError
        
        x = np.arange(start, end, dtype=float)
        if len(x) != len(y):
            return np.inf  # Shape mismatch safeguard
        
        X = np.column_stack([x, np.ones(n, dtype=float)])
        
        try:
            # Try lstsq first
            beta, residuals, rank, s = lstsq(X, y, rcond=None)
            if residuals.size > 0:
                sse = float(residuals.sum())
            else:
                y_hat = X @ beta
                sse = float(np.sum((y - y_hat) ** 2))
        except np.linalg.LinAlgError:
            # Fallback: use mean model if linear fails (collinear points)
            mu = y.mean()
            sse = float(np.sum((y - mu) ** 2))
        
        mse = sse / n if n > 0 else np.inf
        mse = max(mse, 1e-12)
        return n * np.log(mse) + 2 * self.k_params

'''
    def error(self, start, end):
        y = self.signal[start:end].reshape(-1, 1)
        n = end - start
        if n < self.min_size:
            raise ValueError("Segment too short")

        # x = time index; use actual timestamps instead if you have them
        x = np.arange(start, end, dtype=float).reshape(-1, 1)
        X = np.column_stack([x, np.ones_like(x)])  # columns: x, 1

        # Least-squares: y ≈ X @ beta
        beta, residual, _, _ = lstsq(X, y, rcond=None)
        if residual.size == 0:
            # fallback if lstsq doesn't return residuals (e.g. rank issues)
            y_hat = X @ beta
            sse = float(np.sum((y - y_hat) ** 2))
        else:
            sse = float(residual.sum())

        mse = sse / n
        mse = max(mse, 1e-12)  # avoid log(0)

        return n * np.log(mse) + 2 * self.k_params
'''