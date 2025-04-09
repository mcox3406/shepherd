"""
Minimal setup.py for backward compatibility.
For modern package configuration, see pyproject.toml.
"""
import setuptools


if __name__ == "__main__":
    setuptools.setup(
        name="shepherd",
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
    )
