import numpy as np
from ruptures.base import BaseCost

class QRMSELinearCost(BaseCost):
    """
    QRMSE per segment for a linear model:
        y_t ≈ a * t + b
        QRMSE = sqrt(sum(residual^2)) / n
    """
    model = "qrmse_linear"
    min_size = 3  # at least 2 params + 1 extra point

    def fit(self, signal):
        self.signal = np.asarray(signal, dtype=float)
        return self

    def error(self, start, end):
        y = self.signal[start:end]
        n = end - start
        if n < self.min_size:
            raise ValueError("Segment too short")

        # x = time index within the original series
        x = np.arange(start, end, dtype=float)

        # Design matrix for [x, 1]
        X = np.column_stack([x, np.ones_like(x)])

        # Least squares: beta = (X'X)^(-1) X'y
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        residuals = y - y_hat
        sse = np.sum(residuals**2)

        # QRMSE = sqrt(SSE) / n
        return np.sqrt(sse) / n
