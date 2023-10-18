from setuptools import find_packages, setup
from typing import List

# File path for the requirements.txt file
REQUIREMENTS_FILE = 'requirements.txt'
HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Strip the newline character from each requirement
        requirements = [req.replace("\n","") for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    
    return requirements

# Get the list of requirements from the requirements.txt file
requirements = get_requirements(REQUIREMENTS_FILE)

setup(
    name='Tanzania_Water_Well_Functionality',  # Package name
    version='0.0.1',  # Package version
    author='Akshay Pethe',  # Author's name
    author_email='akshay.pethe16@gmail.com',  # Author's email
    install_requires=requirements,  # List of required packages
    packages=find_packages(),  # Automatically find all packages in the project
    )