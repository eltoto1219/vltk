from setuptools import setup


setup(
    name="frcnn",
    version="1",
    author="Antonio Mendoza",
    author_email="avmendoz@cs.unc.edu",
    packages=["frcnn"],
    scripts=["./scripts/checkpoint.sh"],
    description="PyTorch implementation of faster-rcnn",
    long_description=open("README.txt").read(),
    install_requires=[
        "torch",
        "torchvision",
        "pyaml",
        "numpy",
        "pickle",
        "pillow",
        "matplotlib",
        "opencv-python",
        "io",
        "colorsys",
    ],
)
