# your_module.py or tests/test_qlaw_controller.py

import numpy as np
import pytest

from qlaw import QlawController, Qlawgains


class DummySpacecraft:
    def __init__(self, max_thrust):
        self.max_thrust = max_thrust


def test_control_with_real_dependencies():

    spacecraft = DummySpacecraft(max_thrust=5.0)
    controller = QlawController(spacecraft)

    # Example state vector: [x, y, z, vx, vy, vz, mass]
    y = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 1000.0]

    # Your real target should match whatever format cart2eq / qlaw expects
    target = [227939200.0, 0.0934, 0.0, 0.0, 0.0]

    thrust, q = controller.control(
        mu=398600.0,
        y=y,
        target=target,
        Qprev=1.0,
    )
    
    print("thrust", thrust)
    print("Q", q)

    # assert thrust.shape == (3,)
    # assert np.isfinite(q)
    # assert np.linalg.norm(thrust) >= 0.0

if __name__ == "__main__":
    pytest.main(["-s",__file__])