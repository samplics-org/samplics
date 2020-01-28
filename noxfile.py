import nox

LINT_FILES = "docs", "src", "tests", "noxfile.py"


@nox.session(python=["3.6", "3.7", "3.8"])
def tests(session):
    session.install("pytest")
    session.install("-e", ".")
    session.run("pytest", "-v", "tests")


@nox.session
def docs(session):
    session.run("sphinx-build", "-W", "-n", "-b", "html", "docs/source", "docs/build")


@nox.session
def format(session):
    session.install("black", "isort")
    session.run("black", *LINT_FILES)
    session.run("isort", "--recursive", *LINT_FILES)


@nox.session
def lint(session):
    session.install("black", "isort", "mypy")
    session.run("black", "--diff", "--check", *LINT_FILES)
    session.run("isort", "--diff", "--check-only", "--recursive", *LINT_FILES)
    session.run("mypy", "--strict", "--ignore-missing-imports", "src")
    session.run("mypy", "--strict", "--ignore-missing-imports", "-2", "src")


@nox.session
def publish(session):
    session.run("rm", "-rf", "dist", "build", "*.egg-info")
    session.run("poetry", "build")
    session.run("poetry", "publish")
