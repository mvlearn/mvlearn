import os
import sys
from setuptools import setup, find_packages


PACKAGE_NAME = "mvlearn"
DESCRIPTION = "A set of python modules for multiview learning"
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = (
    "Richard Guo",
    "Ronan Perry",
    "Gavin Mischler",
    "Theo Lee",
    "Alexander Chang",
    "Arman Koul",
    "Cameron Franz",
    "Alexandre Gramfort",
    "Hugo Richard",
    "Pierre Ablin",
    "Iain Carmichael",
)
AUTHOR_EMAIL = "rflperry@gmail.com"
URL = "https://github.com/mvlearn/mvlearn"
MINIMUM_PYTHON_VERSION = 3, 6  # Minimum of Python 3.6
REQUIRED_PACKAGES = [
    "matplotlib>=3.0.0",
    "numpy>=1.17.0",
    "scikit-learn>=0.19.1",
    "scipy>=1.5.0",
    "seaborn>=0.9.0",
    "joblib>=0.11",
]

# Find mvlearn version.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "mvlearn", "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


check_python_version()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires=REQUIRED_PACKAGES,
    url=URL,
    project_urls={
      'Documentation': 'https://mvlearn.github.io',
      'Source': URL,
      'Tracker': URL + '/issues/',
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
    packages=find_packages(),
    include_package_data=True,
)
