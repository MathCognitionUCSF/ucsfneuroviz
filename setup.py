from setuptools import setup, find_packages

setup(
    name='ucsfneuroviz',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'nilearn',
        'nibabel',
        'ipywidgets',
        'IPython',
        'os'
    ],
)