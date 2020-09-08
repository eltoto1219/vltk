from subprocess import call

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


class Develop(develop):
    def run(self):
        try:
            call(["./scripts/checkpoint"])
        except Exception:
            print("could not retrieve checkpoint")
        super().run()


class Install(install):
    def run(self):
        try:
            call(["./scripts/checkpoint"])
        except Exception:
            print("could not retrieve checkpoint")
        super().run()


setup(
    name="frcnn",
    version="1.0.0",
    cmdclass={"develop": Develop, "install": Install},
    author="Antonio Mendoza",
    author_email="avmendoz@cs.unc.edu",
    packages=["frcnn"],
    scripts=["scripts/checkpoint"],
    description="PyTorch implementation of faster-rcnn",
    long_description=open("README.md").read(),
    install_requires=[
        "torch",
        "torchvision",
        "pyaml",
        "numpy",
        "pillow",
        "matplotlib",
        "opencv-python",
    ],
)
