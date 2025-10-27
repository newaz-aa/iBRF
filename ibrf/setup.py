# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:11:11 2025

@author: asifn
"""

from setuptools import setup, find_packages

setup(
    name="iBRF",
    version="0.1.0",
    author="Asif Newaz",
    author_email="eee.asifnewaz@iut-dhaka.edu",
    description="Improved Balanced Random Forest (iBRF) Classifier",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/newaz-aa/iBRF",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.2.0",
        "imbalanced-learn>=0.11.0",
        "numpy>=1.23",
        "scipy>=1.10",
        "joblib",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
