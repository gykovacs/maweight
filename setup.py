from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='maweight',
      version='0.1',
      description='Multi-atlas based weight estimation from CT images',
      long_description=readme(),
      classifiers=[
              'Development Status :: 3 - Alpha',
              'License :: GPL3',
              'Programming Language :: Python :: 3.6',
              'Topic :: Image Processing'],
      url='http://github.com/gykovacs/maweight',
      author='Gyorgy Kovacs',
      author_email='gyuriofkovacs@gmail.com',
      license='GPL3',
      packages=find_packages(),
      install_requires=[
              'numpy',
              'pandas',
              'nibabel',
              'imageio',
              'scipy',
              'scikit-learn',
              'xgboost'
              ],
      test_suite='nose.collector',
      tests_require=['nose'],
      scripts=['bin/maweight',
               'bin/imgprop',
               'bin/imgoffset'],
      py_modules=['maweight', 'maweight.mltoolkit'],
      zip_safe=False)
