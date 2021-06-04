def wrap(value, wrapper='"'):
    """Jinja2 map filter to wrap list items in a string on both sides.

    E.g.: ['a', 'b', 'c'] -> ['"a"', '"b"', '"c"']
    """
    return wrapper + value + wrapper
