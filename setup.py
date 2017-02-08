from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    'to install numpy'
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='coclust',
      version='0.2.0',
      description='coclustering algorithms for data mining',
      long_description=readme(),
      classifiers=['Topic :: Scientific/Engineering :: Information Analysis'],
      url='',
      author='Francois Role, Stanislas Morbieu, Mohamed Nadif',
      author_email='francois.role@gmail.com',
      license='BSD3',
      packages=['coclust',
                'coclust/clustering',
                'coclust/coclustering',
                'coclust/evaluation',
                'coclust/io',
                'coclust/visualization'
                ],
      setup_requires=["numpy"],
      install_requires=[
          'numpy', 'scipy', 'scikit-learn'
      ],
      extras_require={
        'alldeps': (
            'numpy',
            'scipy',
            'scikit-learn',
            'matplotlib>=1.5',
            'munkres'
        )
      },
      cmdclass={
          'build_ext': build_ext
      },
      entry_points={
          'console_scripts': [
              'coclust = coclust.coclust:main_coclust',
              'coclust-nb = coclust.coclust:main_coclust_nb',
            ],
      },
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
