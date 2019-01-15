#!/usr/bin/env python

from setuptools import setup, find_packages
import sys
sys.path.append(".")
print(sys.path)

version = '1.0'

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='QSAR_Predictor',
    version=version,
    description='QSAR Predictor for small molecules',
    long_description=readme,
    keywords=['QSAR'],
    author='Steffen Lemke, Thomas Zajac',
    author_email='steffen.lemke@student.uni-tuebingen.de, thomas.zajac@student.uni-tuebingen.de',
    license=license,
    scripts=['scripts/PredictorLemkeZajac.py'],
    install_requires=required,
    packages=find_packages('scripts'),
    include_package_data=True,

)
