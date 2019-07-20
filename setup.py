from setuptools import setup

def readme():
    with open('README.rst') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'trimap',
    'version': '1.0.1',
    'description' : 'TriMap: Dimensionality Reduction Using Triplets',
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
        'Programming Language :: Python :: 2.7',
    ],
    'keywords' : 'dimension reduction triplet t-sne largevis',
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
