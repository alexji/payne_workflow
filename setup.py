#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from setuptools import setup, find_packages
from codecs import open
from os import path, system
from re import compile as re_compile

# For convenience.
if sys.argv[-1] == "publish":
    system("python setup.py sdist upload")
    sys.exit()

def read(filename):
    kwds = {"encoding": "utf-8"} if sys.version_info[0] >= 3 else {}
    with open(filename, **kwds) as fp:
        contents = fp.read()
    return contents

here = path.abspath(path.dirname(__file__))

setup(
    name="payne_workflow",
    version=0.1,
    author="Alex Ji",
    description="Code to streamline using the Payne",
    long_description=read(path.join(here, "README.md")),
    url="https://github.com/alexji/payne_workflow",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    keywords="astronomy",
    packages=find_packages(exclude=["documents", "tests"]),
    install_requires=[
        "numpy","scipy","pyyaml"
        ],
    extras_require={
        #"test": ["coverage"]
    },
    package_data={
        "": ["LICENSE"]
    },
    include_package_data=True,
    data_files=None,
    entry_points=None
)
