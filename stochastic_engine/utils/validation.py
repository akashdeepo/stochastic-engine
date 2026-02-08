"""Input validation utilities."""

from typing import Any


def validate_positive(value: float, name: str) -> None:
    """
    Validate that a value is strictly positive.

    Parameters
    ----------
    value : float
        The value to validate.
    name : str
        Parameter name for error message.

    Raises
    ------
    ValueError
        If value <= 0.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    """
    Validate that a value is non-negative.

    Parameters
    ----------
    value : float
        The value to validate.
    name : str
        Parameter name for error message.

    Raises
    ------
    ValueError
        If value < 0.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_in_range(
    value: float,
    name: str,
    min_val: float | None = None,
    max_val: float | None = None,
    inclusive: bool = True,
) -> None:
    """
    Validate that a value is within a range.

    Parameters
    ----------
    value : float
        The value to validate.
    name : str
        Parameter name for error message.
    min_val : float | None
        Minimum value (if None, no lower bound).
    max_val : float | None
        Maximum value (if None, no upper bound).
    inclusive : bool
        Whether bounds are inclusive.

    Raises
    ------
    ValueError
        If value is outside the range.
    """
    if inclusive:
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")
    else:
        if min_val is not None and value <= min_val:
            raise ValueError(f"{name} must be > {min_val}, got {value}")
        if max_val is not None and value >= max_val:
            raise ValueError(f"{name} must be < {max_val}, got {value}")


def validate_probability(value: float, name: str) -> None:
    """
    Validate that a value is a valid probability (0 to 1).

    Parameters
    ----------
    value : float
        The value to validate.
    name : str
        Parameter name for error message.

    Raises
    ------
    ValueError
        If value is not between 0 and 1.
    """
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def validate_array_shape(
    arr: Any,
    name: str,
    expected_ndim: int | None = None,
    expected_shape: tuple | None = None,
) -> None:
    """
    Validate array dimensions and shape.

    Parameters
    ----------
    arr : array-like
        The array to validate.
    name : str
        Parameter name for error message.
    expected_ndim : int | None
        Expected number of dimensions.
    expected_shape : tuple | None
        Expected shape (use -1 for any size in that dimension).

    Raises
    ------
    ValueError
        If array doesn't match expected dimensions/shape.
    """
    import numpy as np

    arr = np.asarray(arr)

    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(
            f"{name} must be {expected_ndim}D, got {arr.ndim}D"
        )

    if expected_shape is not None:
        for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
            if expected != -1 and actual != expected:
                raise ValueError(
                    f"{name} has wrong shape in dimension {i}: "
                    f"expected {expected}, got {actual}"
                )
