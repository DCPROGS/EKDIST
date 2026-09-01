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
      python_requires='>=3.11',      # dcio's floor, inherited
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'dcio>=0.1.0',      # the shared record layer
          #'PyQt5',
      ],
      zip_safe=False)