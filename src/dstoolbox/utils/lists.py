"""Utility functions for lists and list-likes."""

import itertools


def dict_product(list_dict):
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
