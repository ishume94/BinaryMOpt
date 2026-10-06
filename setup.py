from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent

setup(
    name="bin-mopt",
    version="0.3.0",
    author='Isaac Huidobro',
    author_email='huidobri@mcmaster.ca',
    description="Binary optimization of commuting Pauli measurement groupings.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "networkx>=3.0",
        "numpy>=1.24",
        "openfermion>=1.6",
        "pyscf>=2.4",
        "scipy>=1.9",
        "tequila-basic>=1.9",
        "matplotlib>=3.7",
        "threadpoolctl>=3.1",
        "PySCIPOpt>=5.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
