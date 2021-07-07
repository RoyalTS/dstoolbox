"""Utility functions for lists and list-likes."""

import itertools
import typing


def dict_product(list_dict: typing.Dict[str, list]) -> typing.List[dict]:
    """Create the list-of-dictionary Cartesian product of a dictionary-of-lists.

    Parameters
    ----------
    list_dict : dict of lists

    Returns
    -------
    list
        of dicts


    Example:
    ========
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(list_dict, x)) for x in itertools.product(*list_dict.values()))


# shamelessly pilfered from https://stackoverflow.com/a/60776516/1291563
def dict_replace_value(d: dict, old: typing.Any, new: typing.Any) -> dict:
    """Recursively replace old with new in a dictionary

    Parameters
    ----------
    d : dict
    old : Any
        value to be replaced
    new : Any
        value to replace old with

    Returns
    -------
    dict
    """
    x = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = dict_replace_value(v, old, new)
        elif isinstance(v, list):
            v = list_replace_value(v, old, new)
        elif isinstance(v, str):
            v = v.replace(old, new)
        x[k] = v
    return x


# shamelessly pilfered from https://stackoverflow.com/a/60776516/1291563
def list_replace_value(l: list, old: typing.Any, new: typing.Any) -> list:
    """Recursively replace old with new in a list

    Parameters
    ----------
    d : list
    old : Any
        value to be replaced
    new : Any
        value to replace old with

    Returns
    -------
    list
    """
    x = []
    for e in l:
        if isinstance(e, list):
            e = list_replace_value(e, old, new)
        elif isinstance(e, dict):
            e = dict_replace_value(e, old, new)
        elif isinstance(e, str):
            e = e.replace(old, new)
        x.append(e)
    return x
