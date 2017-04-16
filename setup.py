from distutils.core import setup

setup(
    name = 'shuffled-stats',
    version = '1.0.0',
    description = 'Python library for performing inference on datasets with shuffled labels',
    author = 'Abubakar Abid',
    author_email = 'a12d@stanford.edu',
    url = 'https://github.com/abidlabs/shuffled-stats', 
    py_modules=['shuffled-stats'],
    install_requires=[
        'numpy',
        'scipy',
    ],
    entry_points='''
        [console_scripts]
        zaiste=zaiste:cli
    ''',
)