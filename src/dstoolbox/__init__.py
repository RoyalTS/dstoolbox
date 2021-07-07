from importlib_metadata import distribution, packages_distributions

module_name = __name__
dist_name = packages_distributions()[module_name][0]
__version__ = distribution(dist_name).version
