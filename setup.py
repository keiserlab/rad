from setuptools import setup, find_packages
import os

setup(
    name="rad",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "redis>=5.2.1",
        "matplotlib>=3.5.0",
        "numpy>=1.26.4",
        "tqdm>=4.66.2",
        "notebook>=6.5.0",
        "rdkit>=2024.09.5",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "requests>=2.31.0",
        "pytest>=7.0.0",
        "httpx>=0.24.0",  # Required for FastAPI TestClient
        # TODO: Improve the way we install the usearch submodule 
        f"usearch @ file://localhost/{os.getcwd()}/usearch/"
    ],
    python_requires=">=3.11",
    include_package_data=True,
)