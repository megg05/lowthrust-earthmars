from dataclasses import dataclass

@dataclass
class Body:
    name: str
    mu: float
    r: float

Earth = Body("earth", 3.986e14, 6378137.0)
Mars = Body("mars", 4.283e13, 3389500.0)
Sun = Body("sun", 1.327e20, 6957.0e5)