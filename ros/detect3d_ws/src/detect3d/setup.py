from setuptools import setup, find_packages

setup(
    name='detect3d',
    version='0.0.1',
    packages=find_packages(
        include=[
            'detect3d', 'detect3d.*',
            'heads', 'heads.*',
            'pkgs', 'pkgs.*',
            'BEV',  'BEV.*'
        ]),
    install_requires=['transformations'],
    entry_points={'console_scripts': [
        'detect3d_node = detect3d.detect3d_node:main']},
)
