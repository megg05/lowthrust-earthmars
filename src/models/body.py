from dataclasses import dataclass
import numpy as np
from typing import Callable, Optional, Tuple

@dataclass
class Body:
    name: str
    mu: float
    earth_SOI: float = 9.29e8
    mars_SOI: float = 5.78e8
    a_sun: float = 0.0  # heliocentric semi-major axis [m]

# Earth = Body("earth", 3.986e14, 6378137.0, a_sun=1.496e11)
# Mars = Body("mars", 4.283e13, 3389500.0, a_sun=2.279e11)
# Sun = Body("sun", 1.327e20, 6957.0e5)