from setuptools import setup

version = {}
with open("ttim/version.py") as fp:
    exec(fp.read(), version)

l_d = ""
try:
    import pypandoc

    l_d = pypandoc.convert("README.md", "rst")
except:
    pass


setup(
    name="ttim",
    version=version["__version__"],
    description="Transient multi-layer AEM Model",
    long_description=l_d,
    long_description_content_type='text/markdown',
    author="Mark Bakker",
    author_email="markbak@gmail.com",
    url="https://github.com/mbakker7/ttim",
    license="MIT",
    packages=["ttim"],
    python_requires='>=3.7',
    install_requires=["numpy>=1.17", "scipy>=1.5", "numba>=0.5",
                      "matplotlib>=3.1", "lmfit>=1.0", "pandas>=1.1"],
    classifiers=['Topic :: Scientific/Engineering :: Hydrology'],
)
