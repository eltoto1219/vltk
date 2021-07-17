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
    version="1.0.2",
    # this command is to be used if we want to auto run any scripts
    # cmdclass={"develop": Develop, "install": Install},
    entry_points={"console_scripts": ["vltk = vltk.cli:main"]},
    author="Antonio Mendoza",
    author_email="antonio36764@gmail.com",
    description="The Vision-Language Toolkit (VLTK)",
    long_description=open("README.md").read(),
    packages=[
        "vltk",
        "tests",
        "vltk/utils",
        "vltk/modeling",
        "vltk/modeling/configs",
        "vltk/adapters/",
        "vltk/dataset",
        "vltk/processing",
        "vltk/abc",
    ],
    install_requires=dependencies(),
)
