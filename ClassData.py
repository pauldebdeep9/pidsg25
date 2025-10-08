
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class ProcurementConfig:
    file_path: str
    n_samples: int
    h: float
    b: float
    I_0: float
    B_0: float


@dataclass
class ModelData:
    fixed_demand: np.ndarray
    price_samples: List[Dict[Tuple[int, str], float]]
    order_cost: Dict[str, float]
    lead_time: Dict[str, int]
    capacity_dict: Dict[Tuple[int, str], float]
    T: int
    S: List[str]

