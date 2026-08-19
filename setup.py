from setuptools import find_packages, setup


def readme():
    with open("README.rst") as readme_file:
        return readme_file.read()


configuration = {
    "name": "trimap",
    "version": "1.2.0",
    "description": "TriMap: Large-scale Dimensionality Reduction Using Triplets",
    "long_description": readme(),
    "long_description_content_type": "text/x-rst",
    "classifiers": [
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    "keywords": "Dimensionality Reduction Triplets t-SNE LargeVis UMAP",
    "url": "http://github.com/eamid/trimap",
    "author": "Ehsan Amid",
    "author_email": "eamid@ucsc.edu",
    "license": "Apache-2.0",
    "packages": find_packages(),
    "python_requires": ">=3.9",
    "install_requires": [
        "numpy >= 1.23",
        "scikit-learn >= 1.2",
        "numba >= 0.57",
        "annoy >= 1.17.3",
        "torch >= 2.1",
    ],
    "extras_require": {
        "faiss": ["faiss-cpu >= 1.8"],
        "test": ["pytest >= 7"],
    },
}

setup(**configuration)
