from setuptools import setup, find_packages

setup(
    name="dvsg",
    version="0.1.0",
    description="Tools used for the calculation of DVSG values",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        
    ],
)
