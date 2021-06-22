import sqlalchemy as sa
from snowflake.sqlalchemy import URL
import snowflake.connector
import os


def get_snowflake_credentials_from_env(prefix :str="SNOWFLAKE_", mode:str='sqlalchemy') -> dict:
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

    creds = {
        k.lower().replace(prefix.lower(), ""): v
        for k, v in os.environ.items()
        if k.startswith(prefix)
    }

    if mode == 'sqlalchemy':
        creds['account'] = creds.pop('host')
        creds['user'] = creds.pop('username')

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
    if not 'password' in creds:
        creds['authenticator']="externalbrowser"

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
    if not 'password' in creds:
        creds['authenticator']="externalbrowser"

    engine = sa.create_engine(URL(**creds))

    return engine
