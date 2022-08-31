from setuptools import find_packages, setup

setup(
    name="ray_determined",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["determined", "ray"],
)
