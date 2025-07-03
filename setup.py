from setuptools import find_packages, setup

setup(
    name="src",
    version="0.1.0",
    description=(
        "VoxATtack: A MultiModal Attack on Voice Anonymization Systems"
    ),
    author="Ahmad Aloradi",
    author_email="ahmad.aloradi@fau.de",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(exclude=["tests"]),
)
