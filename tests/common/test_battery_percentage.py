import pytest

from utils.battery import battery_percentage_from_voltage


@pytest.mark.parametrize(
    ("voltage", "expected"),
    [
        (20.0, 0.0),
        (21.0, 0.0),
        (23.1, 0.5),
        (25.2, 1.0),
        (26.0, 1.0),
    ],
)
def test_battery_percentage_from_voltage(voltage, expected):
    assert battery_percentage_from_voltage(voltage) == pytest.approx(expected)
