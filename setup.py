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
    author="Mark Bakker",
    author_email="markbak@gmail.com",
    url="https://github.com/mbakker7/ttim",
    license="MIT",
    packages=["ttim"],
    python_requires='>3.6',
    install_requires=["numpy>=1.17", "scipy>=0.19", "numba>=0.39", "matplotlib>=2.0", "lmfit>=1.0", "pandas>=0.25"],
    classifiers=['Topic :: Scientific/Engineering :: Hydrology'],
)
