"""Month arithmetic for trailing-twelve-month windows. A month is a (year, month) tuple."""
from __future__ import annotations

Month = tuple[int, int]


def shift(month: Month, delta: int) -> Month:
    index = month[0] * 12 + (month[1] - 1) + delta
    return (index // 12, index % 12 + 1)


def window(end: Month, length: int) -> list[Month]:
    """The `length` months ending at (and including) `end`, oldest first."""
    return [shift(end, -i) for i in range(length - 1, -1, -1)]


def label(month: Month) -> str:
    return f"{month[0]}-{month[1]:02d}"
