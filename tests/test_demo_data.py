import numpy as np

from bleve_pressure.demo import generate_scenarios, synthetic_target


def test_demo_generator_is_deterministic_and_positive() -> None:
    first = generate_scenarios(12, seed=7, first_id=1)
    second = generate_scenarios(12, seed=7, first_id=1)

    assert first.equals(second)
    assert first["ID"].is_unique
    assert np.all(synthetic_target(first, seed=8) > 0)
