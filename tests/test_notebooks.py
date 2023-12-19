import os
import shutil
import subprocess
import tempfile

import pytest

nbdirs = [
    # os.path.join("docs/00tutorials"),
    os.path.join("docs/03examples"),
    # os.path.join("notebooks"),
]


testdir = tempfile.mkdtemp()


def get_notebooks():
    nblist = []
    for nbdir in nbdirs:
        nblist += [
            os.path.join(nbdir, f) for f in os.listdir(nbdir) if f.endswith(".ipynb")
        ]
    return nblist


def get_jupyter_kernel():
    try:
        jklcmd = ("jupyter", "kernelspec", "list")
        b = subprocess.Popen(
            jklcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ).communicate()[0]
        if isinstance(b, bytes):
            b = b.decode("utf-8")
        print(b)
        for line in b.splitlines():
            if "python" in line:
                kernel = line.split()[0]
    except:
        kernel = None

    return kernel


@pytest.mark.notebooks
@pytest.mark.parametrize("fn", get_notebooks())
def test_notebook(fn):
    kernel = get_jupyter_kernel()
    print("available jupyter kernel {}".format(kernel))

    cmd = (
        "jupyter "
        + "nbconvert "
        + "--ExecutePreprocessor.timeout=600 "
        + "--to "
        + "notebook "
        + "--execute "
        + "{} ".format(fn)
        + "--output-dir "
        + "{} ".format(testdir)
        + "--output "
        + "{}".format(fn)
    )
    ival = os.system(cmd)
    assert ival == 0, "could not run {}".format(fn)


if __name__ == "__main__":
    test_notebook()
    shutil.rmtree(testdir)
