from subprocess import call

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


# anything we want to run in these classes we can, and setup.py will execute them
class Develop(develop):
    def run(self):
        try:
            call([""])
        except Exception:
            print("")
        super().run()


class Install(install):
    def run(self):
        try:
            call([""])
        except Exception:
            print("")
        super().run()


install_requires = [
    "torch",
    "torchvision",
    "pyaml",
    "numpy",
    "pillow",
    "matplotlib",
    "opencv-python",
    "requests",
    "tqdm",
    "wget",
    "filelock",
    "transformers @ git+https://github.com/huggingface/transformers.git",
    "datasets @ git+https://github.com/huggingface/datasets.git",
    "jupyter",
    "pynvml"
]
setup(
    name="mllib",
    version="1.0.0",
    # cmdclass={"develop": Develop, "install": Install},
    author="Antonio Mendoza",
    author_email="avmendoz@cs.unc.edu",
    description="PyTorch implementation of faster-rcnn",
    long_description=open("README.md").read(),
    packages=["mllib", "mllib/tests"],
    install_requires=install_requires,
),
