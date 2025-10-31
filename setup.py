from setuptools import setup, find_packages

setup(
    name="dvsg",
    version="0.1.0",
    description="Tools used for the calcualtion of DVSG values",
    author="Jonah Powley",
    packages=find_packages(),
    install_requires=[
        "astropy"
    ],
    python_requires=">=3.6",
)