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

setup(name='test0104',
      version='0.1',
      description='coclustering algorithms for data mining',
      long_description=readme(),
      classifiers=['Topic :: Scientific/Engineering :: Information Analysis'],
      url='https://github.com/franrole/cclust_package.git',
      author='Francois Role, Stanislas Morbieu',
      author_email='francois.role@gmail.com',
      license='BSD3',
      packages=['coclust', 'coclust/tests', 'coclust/utils'],
      setup_requires=["numpy"],
      install_requires=[
          'numpy', "scipy", "scikit-learn"
      ],
      cmdclass={
          'build_ext': build_ext
      },
      entry_points={
          'console_scripts': [
              'coclust = coclust.coclust:main_coclust',
              'coclust-nb = coclust.coclust:main_coclust_nb',
              'coclust-demo = coclust.coclust_demo:main_coclust_demo',
            ],
      },
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
