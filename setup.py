from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("src/spey_hs3/_version.py", encoding="UTF-8") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = ["pyhs3", "spey>=0.2.1"]

docs = [
    "sphinx==6.2.1",
    "sphinxcontrib-bibtex~=2.1",
    "sphinx-click",
    "sphinx_rtd_theme",
    "nbsphinx!=0.8.8",
    "sphinx-issues",
    "sphinx-copybutton>=0.3.2",
    "sphinx-togglebutton>=0.3.0",
    "myst-parser",
    "sphinx-rtd-size",
]

setup(
    name="spey-hs3",
    version=version,
    description=("HS3 plugin for spey interface"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpeysideHEP/spey-hs3",
    project_urls={
        "Bug Tracker": "https://github.com/SpeysideHEP/spey-hs3/issues",
        "Documentation": "https://spey-hs3.readthedocs.io",
        "Repository": "https://github.com/SpeysideHEP/spey-hs3",
        "Homepage": "https://github.com/SpeysideHEP/spey-hs3",
        "Download": f"https://github.com/SpeysideHEP/spey-hs3/archive/refs/tags/v{version}.tar.gz",
    },
    download_url=f"https://github.com/SpeysideHEP/spey-hs3/archive/refs/tags/v{version}.tar.gz",
    author="Jack Y. Araz",
    author_email=("j.araz@ucl.ac.uk"),
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={"spey.backend.plugins": ["hs3 = spey_hs3:HS3Interface"]},
    install_requires=requirements,
    python_requires=">=3.8, <3.14",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    extras_require={
        "dev": ["pytest>=7.1.2", "pytest-cov>=3.0.0", "twine>=3.7.1", "wheel>=0.37.1"],
        "doc": docs,
    },
)
