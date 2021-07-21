from setuptools import setup

import versioneer

if __name__ == "__main__":
    try:
        setup(
            version=versioneer.get_version(),
            cmdclass=versioneer.get_cmdclass(),
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n",
        )
        raise
