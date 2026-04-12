from dataclasses import dataclass, field

@dataclass
class SolverConfig:
    num_nodes: int = 60
    max_iter: int = 500
    tol: float = 1e-8
    method: str = "SLSQP"