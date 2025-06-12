from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Returns a list of dependencies from the given requirements file.
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
    requirements = [req.strip() for req in requirements if req.strip()]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="Apoorv",
    author_email="apoorvsahu.cse19@chitkarauniversity.edu.in",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements("requirements.txt")
)
