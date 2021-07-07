import pandas as pd


def set_output_defaults(
    max_rows: int = 999,
    max_columns: int = 999,
    decimals: int = 4,
) -> None:
    """Set output defaults for pandas

    Parameters
    ----------
    max_rows : int, optional
        Number of rows to display, by default 999
    max_columns : int, optional
        Number of columns to display, by default 999
    decimals : int, optional
        Number of decimals to display for floating point numbers, by default 4
    """
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", max_columns)

    format_string = "{:." + str(decimals) + "f}"
    pd.set_option("display.float_format", format_string.format)
