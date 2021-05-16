"""git_root() module."""

import subprocess
from pathlib import Path, PurePath


def git_root() -> PurePath:
    """Return the root path of the git repository.

    Returns
    -------
    PurePath
        path to the root of the git repository

    Example
    -------
    > git_root()
    /home/user/my_repo
    """
    git_root = (
        subprocess.Popen(
            ["git", "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
        )
        .communicate()[0]
        .rstrip()
        .decode("utf-8")
    )
    return Path(git_root)
