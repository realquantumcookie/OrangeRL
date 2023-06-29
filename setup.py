# setup.py
import os
from setuptools import setup, find_packages

def read_requirements_file(filename):
    req_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]

setup(
    name='orangerl',
     version='0.0.1a',
    description="ROAR_PY interface definitions and streaming capabilities library",
    url="https://github.com/realquantumcookie/OrangeRL",
    classifiers=[
        "Programming Language :: Python :: 3",
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    author="Yunhao Cao",
    keywords=["reinforcement learning", "gymnasium"],
    license="MIT",
    install_requires=read_requirements_file("requirements.txt"),
    packages=['orangerl'], # find_packages(),
    python_requires='>=3.10',
)