from setuptools import setup, find_packages

setup(
    name="shepherd",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    description="ShEPhERD: Diffusing Shape, Electrostatics, and Pharmacophores for Drug Design",
    author="Keir Adams and Kento Abeywerdane",
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torch-geometric",
        "pytorch-lightning",
        "rdkit",
        "e3nn",
        "open3d",
        "xtb-python"
    ],
)