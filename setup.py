from setuptools import setup

setup(name='ekdist',
      version='0.0.0',
      description="EKDIST- explore single channel intervals",
      url='https://github.com/DCPROGS/EKDIST',
      keywords='histogram fit exponential pdf gaussian',
      author='Remis Lape',
      author_email='',
      license='MIT',
      packages=['ekdist'],
      install_requires=[
          'numpy',
          'matplotlib',
          # The shared record layer. Not on PyPI yet, so a source install
          # needs it first:
          #   pip install git+https://github.com/DCPROGS/dcio
          'dcio',
          #'PyQt5',
      ],
      zip_safe=False)