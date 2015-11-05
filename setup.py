from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='coclust',
      version='0.2',
      description='coclustering algorithms',
      long_description=readme(),
      classifiers=['Topic :: Data Mining :: Co-clustering'],
      url='https://github.com/franrole/cclust_package.git',
      author='XXXX',
      author_email='XXXX',
      license='BSD3',
      packages=['coclust' ,'coclust/tests', 'coclust/utils'],
      setup_requires=[],
      install_requires=[
          'numpy',"scipy","nose", "scikit-learn"
      ],
##      entry_points={
##          'console_scripts': [
##              'coclust = bin.coclust:main',
##            ],
##      },
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      scripts=['bin/coclust.py'],
      )
