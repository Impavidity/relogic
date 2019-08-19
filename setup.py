from setuptools import find_packages, setup

setup(
  name='relogic',
  version='0.0.1',
  author='Peng Shi',
  author_email="peng_shi@outlook.com",
  description='Natural Language Processing Toolkits',
  url='https://github.com/Impavidity/relogic',
  license='MIT',
  install_requires=['boto3', 'Cython', 'spacy', 'pyjnius'],
  package_data={},
  packages=find_packages(),
)
