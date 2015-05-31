from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='coclust',
      version='0.2',
      description='coclustering algorithms',
      url='http://github.com/',
      author='XXXX',
      author_email='XXXX',
      license='MIT',
      packages=['coclust'],
      setup_requires=["numpy","scipy"],
      install_requires=[
          'numpy',"scipy"
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      scripts=['bin/launch_coclust'],)
