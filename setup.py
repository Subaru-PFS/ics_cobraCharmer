from setuptools import setup, find_packages

setup(name="ics.cobraCharmer",
      #version="x.y",
      author="Craig Loomis",
      #author_email="",
      #description="",
      url="https://github.com/Subaru-PFS/ics_cobraCharmer/",
      packages=find_packages('python'),
      package_dir={'':'python'},
      zip_safe=False,
      license="GPLv3",
      install_requires=["numpy", "matplotlib"],
      )
