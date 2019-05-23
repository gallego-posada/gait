from setuptools import setup, find_packages


with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

setup(
    name='pylego',
    version='0.1.0_dev',
    description='Tools for writing extendable machine learning code.',
    author='Anonymous',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    author_email='anon@anon.com',
)
