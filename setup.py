from setuptools import setup

def readme():
    with open('README.rst') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'trimap',
    'version': '1.0.9',
    'description' : 'TriMap: Large-scale Dimensionality Reduction Using Triplets',
    'long_description' : readme(),
    'classifiers' : [
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.6',
    ],
    'keywords' : 'Dimensionality Reduction Triplets t-SNE LargeVis UMAP',
    'url' : 'http://github.com/eamid/trimap',
    'author' : 'Ehsan Amid',
    'author_email' : 'eamid@ucsc.edu',
    'license' : 'LICENSE.txt',
    'packages' : ['trimap'],
    'install_requires' : ['scikit-learn >= 0.16',
                          'numba >= 0.34',
                          'annoy >= 1.11']
    }

setup(**configuration)
