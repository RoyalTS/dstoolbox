import subprocess


def git_tag() -> str:
    """Get the latest tag of the active git repo.
    Fall back to the latest commit hash if no tag exists

    Returns
    -------
    str
        git tag or commit hash
    """
    return (
        subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
    )
