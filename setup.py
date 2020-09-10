from subprocess import call

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


# anything we want to run in these classes we can, and setup.py will execute them
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
    # cmdclass={"develop": Develop, "install": Install},
    author="Antonio Mendoza",
    author_email="avmendoz@cs.unc.edu",
    # package_dir={"": "src"},
    scripts=["scripts/checkpoint"],
    description="PyTorch implementation of faster-rcnn",
    long_description=open("README.md").read(),
    packages=find_packages(),
)

install_requires = [
    "torch",
    "torchvision",
    "pyaml",
    "numpy",
    "pillow",
    "matplotlib",
    "opencv-python",
    "requests",
    "fnmatch",
    "tqdm",
    "wget",
    "reguests",
]
