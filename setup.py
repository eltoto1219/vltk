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
            l = l.split("==")[0]
            deps.append(l)
    return deps


setup(
    name="mllib",
    version="1.0.0",
    # this command is to be used if we want to auto run any scripts
    # cmdclass={"develop": Develop, "install": Install},
    entry_points={"console_scripts": ["run = mllib.cli:main"]},
    author="Antonio Mendoza",
    author_email="avmendoz@cs.unc.edu",
    description="Personal AI library for quick projects",
    long_description=open("README.md").read(),
    packages=[
        "mllib",
        "tests",
        "mllib/models",
        "mllib/legacy",
        "mllib/processors",
    ],
    install_requires=dependencies(),
)
