from setuptools import setup, find_packages

setup(
    name="rad",
    version="0.2.0",
    packages=find_packages(include=["usearch", "rad"]),
    install_requires=[
        "redis>=5.2.1",
    ],
    python_requires=">=3.11",
    include_package_data=True,
)
