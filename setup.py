from subprocess import call

from setuptools import setup
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


def dependencies() -> list:
    deps = []
    with open("./requirements.txt") as f:
        for l in f.readlines():
            if "-e" in l:
                continue
            deps.append(l)
    return deps


setup(
    name="vltk",
    version="1.0.0",
    # this command is to be used if we want to auto run any scripts
    # cmdclass={"develop": Develop, "install": Install},
    entry_points={"console_scripts": ["run = vltk.cli:main"]},
    author="Antonio Mendoza",
    author_email="avmendoz@cs.unc.edu",
    description="The Vision-Language Toolkit (VLTK)",
    long_description=open("README.md").read(),
    packages=[
        "vltk",
        "tests",
        "vltk/modeling",
        "vltk/legacy",
        "vltk/processing",
        "vltk/exp",
        "vltk/loop",
        "vltk/abc",
        "vltk/textset",
        "vltk/imageset",
        "vltk/maps"
        "vltk/modeling/configs"
    ],
    install_requires=dependencies(),
)
