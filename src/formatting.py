"""Deutsche Zahlenanzeige, unabhängig vom Zahlenformat der gespeicherten Daten."""


def format_number(value: float, decimals: int | None = None) -> str:
    """Verwende ein Dezimalkomma, optional mit fester Zahl an Nachkommastellen."""
    spec = "g" if decimals is None else f".{decimals}f"
    return format(value, spec).replace(".", ",")
