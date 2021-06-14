import pandas as pd


def attach_timezone_to_datetime_cols(
    df: pd.DataFrame, tz: str = "UTC"
) -> pd.DataFrame:
    """Attaches a timezone to all datetime64 columns in the DataFrame.

    Writes to Snowflake using pandas' to_sql() method will result in invalid timestamps unless a timezone is attached

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame
    tz : str, optional
        timezone name, by default "UTC"

    Returns
    -------
    pd.DataFrame
    """
    cols = df.select_dtypes(include=["datetime64"]).columns
    for col in cols:
        df[col] = df[col].dt.tz_localize(tz)

    return df