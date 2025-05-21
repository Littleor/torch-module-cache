import os

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torch-module-cache",
    version="0.1.4",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
    ],
    author="Littleor",
    author_email="me@littleor.cn",
    description="A package for caching PyTorch modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Littleor/torch-module-cache",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    license="MIT",
)
