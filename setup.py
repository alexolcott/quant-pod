"""Only exists to declare the pybind11 C++ extension module; everything else
(metadata, dependencies) lives in pyproject.toml.
"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "quant_pod.trader.hotpath._hotpath",
        [
            "quant_pod/trader/hotpath/orderbook.cpp",
            "quant_pod/trader/hotpath/bindings.cpp",
        ],
        cxx_std=17,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
