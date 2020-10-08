from setuptools import setup

setup(
    name='Marc_Guppy',
    url='https://github.com/marc131183/marc_guppy',
    author='Marc Gröling',
    packages=['marc_guppy'],
    install_requires=['numpy', 'stable_baselines', 'scipy'],
    version='0.1',
    description='Custom Guppy with rl model',
)
