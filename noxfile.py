import nox


@nox.session(python=["3.6", "3.7", "3.8"])
def tests(session):
    session.run("poetry", "install")
    session.run("pytest", "-v", "tests")


@nox.session
def docs(session):
    session.install(".")
    session.install(
        "sphinx", "sphinx-autobuild", "sphinx_bootstrap_theme", "nbsphinx", "recommonmark",
    )
    session.chdir("docs")
    # session.run("rm", "-rf", "build/", external=True)
    session.run("make", "clean")
    session.run("sphinx-apidoc", "-o", "source", "../src/samplics")
    sphinx_args = ["-b", "html", "source", "build"]

    if "serve" in session.posargs:
        session.run("sphinx-autobuild", *sphinx_args)
    else:
        session.run("sphinx-build", *sphinx_args)


@nox.session
def lint(session):
    lint_files = ["docs", "src", "tests", "noxfile.py"]
    session.install("black", "mypy")
    session.run("black", "--diff", "--check", *lint_files)
    session.run("mypy", "--strict", "--ignore-missing-imports", "src")


@nox.session
def publish(session):
    session.run("poetry", "install")
    session.run("rm", "-rf", "dist", "build", "*.egg-info")
    session.run("poetry", "build")
    session.run("poetry", "publish")
