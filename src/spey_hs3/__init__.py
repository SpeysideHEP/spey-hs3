import pyhs3
import spey
from ._version import __version__


class HS3Interface(spey.BackendBase):
    name = "hs3"
    version = __version__
    spey_requires = ">=0.2.0"

    def __init__(self, model_file: dict, distribution: str):
        pass

    @property
    def is_alive(self):
        return True

    def config(self, allow_negative_signal: bool = True, poi_upper_bound: float = 10.0):
        pass

    def get_logpdf_func(self, expected=spey.ExpectationType.observed, data=None):

        pass
