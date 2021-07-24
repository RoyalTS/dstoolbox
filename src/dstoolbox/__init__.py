from ._version import get_versions

v = get_versions()
__version__ = v.get("closest-tag", v["version"])
__git_version__ = v.get("full-revisionid")
del get_versions, v

from . import _version
__version__ = _version.get_versions()['version']
