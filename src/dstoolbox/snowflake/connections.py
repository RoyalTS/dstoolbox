from pprint import pformat

import environs
import snowflake.connector
import sqlalchemy as sa
from loguru import logger
from snowflake.sqlalchemy import URL


def get_snowflake_credentials_from_env(
    prefix: str = "SNOWFLAKE_",
    mode: str = "sqlalchemy",
) -> dict:
    """Get snowflake credentials from environment variables

    Parameters
    ----------
    prefix : str, optional
        a prefix common to all environment variables, by default "SNOWFLAKE_"
    mode : str
        either 'sqlalchemy' or 'roo_data_storage'
    Returns
    -------
    dict
        dictionary containing corresponding to the environment variables
        stripped of the prefix
    """

    env = environs.Env()
    env.read_env()

    creds = {}

    with env.prefixed(prefix=prefix):
        creds["host"] = env("HOST", "deliveroo.eu-central-1")
        creds["username"] = env("USERNAME")
        try:
            creds["password"] = env("PASSWORD")
        except environs.EnvError:
            pass
        creds["warehouse"] = env("WAREHOUSE")
        creds["role"] = env("ROLE")
        creds["database"] = env("DATABASE")
        creds["schema"] = env("SCHEMA")

    # SSO if no password supplied
    if "password" not in creds:
        creds["authenticator"] = "externalbrowser"

    # these are the argument names sqlalchemy uses
    if mode == "sqlalchemy":
        if "host" in creds and (not "account" in creds):
            creds["account"] = creds.pop("host")
        if "username" in creds and (not "user" in creds):
            creds["user"] = creds.pop("username")

    # log sanitize credentials
    clean_creds = creds.copy()
    if "password" in clean_creds:
        clean_creds["password"] = "*****"
    logger.debug(
        f"Read in Snowflake credentials from the entironment:\n{pformat(clean_creds)}",
    )

    return creds


def create_snowflake_connection(creds: dict):
    """
    Create a Snowflake connector connection from credentials

    Parameters
    ----------
    creds : dict
        dictionary of credentials containing keys 'user', 'account',
        'warehouse', 'role', 'schema', 'database' and optionally 'password'. If
        no password is supplied authentication via SSO is triggered
    """
    if not "password" in creds:
        creds["authenticator"] = "externalbrowser"

    ctx = snowflake.connector.connect(**creds)

    return ctx


def create_snowflake_engine(creds: dict) -> sa.engine.base.Engine:
    """
    Create an SQLAlchemy engine from credentials

    Parameters
    ----------
    creds : dict
        dictionary of credentials containing keys 'user', 'account',
        'warehouse', 'role', 'schema', 'database' and optionally 'password'. If
        no password is supplied authentication via SSO is triggered

    Returns
    -------
    sa.engine.base.Engine
        SQLAlchemy engine
    """
    if not "password" in creds:
        creds["authenticator"] = "externalbrowser"

    engine = sa.create_engine(URL(**creds))

    return engine
