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


setup(
    name="mllib",
    version="1.0.0",
    # this command is to be used if we want to auto run any scripts
    # cmdclass={"develop": Develop, "install": Install},
    entry_points={"console_scripts": ["run = cli.cli:main"]},
    author="Antonio Mendoza",
    author_email="avmendoz@cs.unc.edu",
    description="Personal AI library for quick projects",
    long_description=open("README.md").read(),
    packages=["cli", "mllib", "mllib/tests"],
    install_requires=[],
)
