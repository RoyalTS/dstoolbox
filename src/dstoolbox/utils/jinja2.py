import jinja2
import typing
import pathlib
from loguru import logger
import roo_data_storage.snowflake_utils
import pandas as pd


def set_up_template_env(path: typing.Union[str, pathlib.Path]) -> jinja2.Environment:
    """Set up a jinja2 template environment based on a path

    Parameters
    ----------
    path : typing.Union[str, pathlib.Path]
        path

    Returns
    -------
    jinja2.Environment
    """
    template_loader = jinja2.FileSystemLoader(searchpath=path)
    template_env = jinja2.Environment(loader=template_loader)
    template_env.filters["wrap"] = wrap

    return template_env


def execute_template(
    template_file: typing.Union[str, pathlib.Path],
    parameters: dict,
    template_env: jinja2.Environment,
    snow: roo_data_storage.snowflake_utils.RooSnowflakeAdapter,
    coerce: bool = True,
    dump_query: bool = True,
) -> pd.DataFrame:
    """ake the jinja2-templated query in an sql file and execute it against Snowflake.

    Parameters
    ----------
    template_file : typing.Union[str, pathlib.Path]
        Filename of the template to be rendered
    parameters : dict
        dictionary of parameters to be used in rendering
    template_env : jinja2.Environment
        jinja2 environment
    snow : roo_data_storage.snowflake_utils.RooSnowflakeAdapter
        SnowflakeAdapter to run the query against
    coerce : bool, optional
        passed on to SnowflakeAdapter.read* methods
    dump_query : bool, optional
        Dump the rendered query to disk before execution for debugging purposes, by default True

    Returns
    -------
    pd.DataFrame
        Result set of the rendered query
    """
    import time
    import humanize

    template = template_env.get_template(template_file)

    rendered_template = template.render(**parameters)

    # write the query out to disk for debugging before executing it
    if dump_query:
        query_dump_file = pathlib.Path(".query_dump.sql")
        logger.trace(f"Dumping full SQL query to {query_dump_file}")
        with open(query_dump_file, "w") as f:
            f.write(rendered_template)

    logger.debug(f"executing query for template {template_file}")
    logger.trace(f"parameters: {parameters}")
    logger.trace(rendered_template)

    start_time = time.time()

    df = snow.read_data(rendered_template, coerce=coerce)

    execution_time = humanize.naturaldelta(time.time() - start_time)
    logger.debug(f"Query execution time: {execution_time}")

    # delete the query dump after successful execution
    if dump_query:
        logger.trace(f"Deleting SQL query dump at {query_dump_file}")
        query_dump_file.unlink()

    return df


def wrap(value, wrapper='"'):
    """Jinja2 map filter to wrap list items in a string on both sides.

    E.g.: ['a', 'b', 'c'] -> ['"a"', '"b"', '"c"']
    """
    return wrapper + value + wrapper
