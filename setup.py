from setuptools import find_packages, setup
from typing import List

requirements_file_name = "requirements.txt"
REMOVE_PACKAGE = "-e ."

def get_requirements()->List[str]:
    with open(requirements_file_name) as requirement_file:
        requirement_list = requirement_file.readlines()
    requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]

    if REMOVE_PACKAGE in requirement_list:
        requirement_list.remove(REMOVE_PACKAGE)

    return requirement_list

setup(
    name='Insurace',
    version='0.1.0',
    description='Insurance Industry Level Project',
    author='Saurabh Sharma',
    author_email='saurabh.sharma7310@gmail.com',
    # url='https://github.com/Ahmed1234567890',
    packages=find_packages(),
    install_requires = get_requirements())