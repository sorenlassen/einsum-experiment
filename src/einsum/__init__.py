# export the version number.
from importlib.metadata import version
__version__ = version(__name__)
del version

# export einsum API.
from .lib import *

