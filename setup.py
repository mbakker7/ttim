from __future__ import division, absolute_import, print_function
import sys
import os
import platform

version = {}
with open("ttim/version.py") as fp:
    exec(fp.read(), version)

l_d = ""
try:
    import pypandoc

    l_d = pypandoc.convert("README.md", "rst")
except:
    pass

try:
    from numpy.distutils.core import Extension, setup
except ImportError:
    sys.exit("install requires: 'numpy'.")

cputune = ["-march=native"]

if os.name == "nt":
    compile_args = ["-static-libgcc", "-Wall", "-shared"]
else:
    compile_args = ["-static-libgcc", "-Wall", "-lgfortran", "-lquadmath"]
    cputune = []

bessel_ext = Extension(
    name="ttim.bessel",
    sources=["ttim/src/bessel.f95"],
    extra_compile_args=compile_args + cputune,
)
invlap_ext = Extension(
    name="ttim.invlap",
    sources=["ttim/src/invlap.f90"],
    extra_compile_args=compile_args + cputune,
)


def setup_package():

    metadata = dict(
        name="ttim",
        version=version["__version__"],
        description="Transient multi-layer analytical element model",
        long_description=l_d,
        author="Mark Bakker",
        author_email="markbak@gmail.com",
        url="https://github.com/mbakker7/ttim",
        license="MIT",
        packages=["ttim"],
        ext_modules=[bessel_ext, invlap_ext],
    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
