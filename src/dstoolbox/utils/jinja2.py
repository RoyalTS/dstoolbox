import jinja2
import typing
import pathlib
from loguru import logger


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


def wrap(value, wrapper='"'):
    """Jinja2 map filter to wrap list items in a string on both sides.

    E.g.: ['a', 'b', 'c'] -> ['"a"', '"b"', '"c"']
    """
    return wrapper + value + wrapper
