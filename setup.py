try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
    __author__ = 'Jason Koh'
    __version__ = '0.1'
    setup(
            name        = 'crfsharp',
            version     = __version__,
            packages    = ['crfsharp'],
            author      = __author__,
            description = 'A Python wrapper around CRFSharp binary',
            zip_safe    = False, 
            install_requires = ['setuptools'],
            include_package_data = True,
            classifiers = (
                'Development Status :: 1 - Planning',
                'Intended Audience :: Developers'
                ),
            data_files = ('bin/*.exe')
            )
