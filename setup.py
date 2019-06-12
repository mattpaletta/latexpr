from distutils.command.build import build

from setuptools import setup, find_packages

from setuptools_antlr.command import AntlrCommand


class BuildCommand(build):
    def run(self):
        # AntlrCommand.run(build_py)
        build.run(self)

setup(
    name='latexpr',
    version='0.1.1',
    description='Python function generator from LaTeX expressions.',
    maintainer='Matthew Paletta',
    maintainer_email='mattpaletta@gmail.com',
    url='https://github.com/mattpaletta/latexpr',
    packages=find_packages(),
    setup_requires=["setuptools-antlr==0.4.0"],
    install_requires=['antlr4-python3-runtime==4.7.2',
                      'sympy'],
    cmdclass = {
          "build": BuildCommand,
    }
)