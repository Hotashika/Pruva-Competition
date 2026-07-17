"""Batarya voltaji ve doluluk orani donusumleri."""

BATTERY_EMPTY_VOLTAGE = 21.0
BATTERY_FULL_VOLTAGE = 25.2


def battery_percentage_from_voltage(voltage):
    """6S toplam voltajini 21.0-25.2 V arasinda 0.0-1.0'a cevirir."""
    voltage = float(voltage)
    ratio = (
        (voltage - BATTERY_EMPTY_VOLTAGE)
        / (BATTERY_FULL_VOLTAGE - BATTERY_EMPTY_VOLTAGE)
    )
    return max(0.0, min(1.0, ratio))
