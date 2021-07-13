import typing
from datetime import date, datetime

import pandas as pd
import sqlalchemy


def attach_timezone_to_datetime_cols(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
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


def get_last_commit_time(object_name, engine):
    with engine.connect() as con:
        result = con.execute(
            f"SELECT TO_TIMESTAMP(SYSTEM$LAST_CHANGE_COMMIT_TIME('{object_name}') / 1000)",
        ).fetchone()[0]

    return result


def check_is_fresh(
    object_name: str,
    engine: sqlalchemy.engine.base.Engine,
    same_day: bool = True,
    max_hours: typing.Union[int, None] = None,
) -> bool:
    """Check if the Snowflake object was last modified today and/or less than max_hours ago."""
    last_modified = get_last_commit_time(object_name, engine)

    if same_day:
        same_day_satisfied = last_modified.date() == date.today()
    else:
        same_day_satisfied = True

    if max_hours:
        secs_since_modified = (datetime.now() - last_modified).seconds
        max_hours_satisfied = secs_since_modified / 60 / 60 < max_hours
    else:
        max_hours_satisfied = True

    both_satisfied = max_hours_satisfied and same_day_satisfied

    return both_satisfied


class NotFreshError(Exception):
    """Exception raised when some Snowflake objects are not fresh

    Attributes
    ----------
    objects : names of the objects that are not fresh
    message : The message displayed
    """

    def __init__(self, objects, message="Some objects are not fresh"):
        self.objects = objects
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}: {self.objects}"


def raise_if_not_all_fresh(
    objects: typing.List[str],
    engine=sqlalchemy.engine.base.Engine,
    same_day: bool = False,
    max_hours: typing.Union[int, None] = None,
) -> None:
    """Raise ValueError unless all Snowflake objects in objects were last modified today and/or less than max_hours ago."""
    if not same_day and not max_hours:
        raise ValueError("You must set either same_day or max_hours")
    not_fresh = {
        obj: f"{get_last_commit_time(obj, engine):%Y-%m-%d %H:%M}"
        for obj in objects
        if not check_is_fresh(obj, engine, same_day=same_day, max_hours=max_hours)
    }
    if len(not_fresh) > 0:
        raise NotFreshError(objects=not_fresh)


def check_non_empty(object_name: str, engine: sqlalchemy.engine.base.Engine):
    """Check if a Snowflake object has a COUNT() > 0."""
    with engine.connect() as con:
        count = con.execute(f"SELECT COUNT(*) FROM {object_name}").fetchone()[0]

    return count > 0


class EmptyError(Exception):
    """Exception raised when some Snowflake objects are empty.

    Attributes
    ----------
    objects : names of the objects that are empty
    message : The message displayed
    """

    def __init__(self, objects, message="Some objects are empty"):
        self.objects = objects
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}: {self.objects}"


def raise_if_any_empty(
    objects: typing.List[str],
    engine=sqlalchemy.engine.base.Engine,
):
    """Raise ValueError if any passed Snowflake objects are empty."""
    empty = [obj for obj in objects if not check_non_empty(obj, engine)]
    if len(empty) > 0:
        raise EmptyError(objects=empty)


def warehouse_info(
    warehouse_name: str,
    engine: sqlalchemy.engine.base.Engine,
) -> pd.Series:
    """Query for information about a warehouse

    Parameters
    ----------
    warehouse_name : str
        name of the warehouse
    engine : sqlalchemy.engine.base.Engine
        sqlalchemy

    Returns
    -------
    pd.Series
        Series containing information about the warehouse
    """
    warehouse_info = pd.read_sql(
        f"SHOW WAREHOUSES LIKE '{warehouse_name}'",
        engine,
    )[["name", "type", "size", "started_clusters", "running", "queued"]]

    return warehouse_info.iloc[0]
