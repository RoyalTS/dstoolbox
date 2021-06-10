import sqlalchemy as sa
from snowflake.sqlalchemy import URL
import snowflake.connector
import os


def get_snowflake_credentials_from_env(prefix :str="SNOWFLAKE_DB_") -> dict:
    """Get snowflake credentials from environment variables

    Parameters
    ----------
    prefix : str, optional
        a prefix common to all environment variables, by default "SNOWFLAKE_DB_"

    Returns
    -------
    dict
        dictionary containing corresponding to the environment variables
        stripped on the prefix
    """

    creds = {
        k.lower().replace(prefix.lower(), ""): v
        for k, v in os.environ.items()
        if k.startswith(prefix)
    }

    creds["database"] = creds.pop("name")

    return creds


def create_snowflake_connection(creds: dict):
    """
    Create a Snowflake connector connection from credentials

    Parameters
    ----------
    creds : dict
        dictionary of credentials containing keys 'user', 'account', 'warehouse', 'role', 'schema', 'database'
    """

    ctx = snowflake.connector.connect(authenticator="externalbrowser", **creds)

    return ctx


def create_snowflake_engine(creds: dict) -> sa.engine.base.Engine:
    """
    Create an SQLAlchemy engine from credentials

    Parameters
    ----------
    creds : dict
        dictionary of credentials containing keys 'user', 'account', 'warehouse', 'role', 'schema', 'database'

    Returns
    -------
    sa.engine.base.Engine
        SQLAlchemy engine
    """
    engine = sa.create_engine(URL(**creds, authenticator="externalbrowser"))

    return engine
