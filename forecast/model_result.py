from dataclasses import dataclass
from typing import Any, Optional

# per avere l'output di tutti i modelli coerenti
@dataclass
class ModelResult:
    aic: float
    rmse: float
    model: tuple
    seasonal_model: tuple
    params: Optional[dict] = None
    fitted: Optional[Any] = None