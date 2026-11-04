# Contributing to GroMo

Thank you for your interest in GroMo! Contributions of all kinds are welcome:
bug reports, questions, documentation, examples and code.

By participating in this project, you agree to follow our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Questions, bugs and ideas

- Questions and ideas: open a thread in
  [Discussions](https://github.com/growingnet/gromo/discussions).
- Bugs and feature requests: open an
  [issue](https://github.com/growingnet/gromo/issues/new/choose) with one of the
  templates. For a bug, give a minimal example that reproduces it, and the versions
  of gromo, PyTorch and Python.
- Security vulnerabilities: do not open a public issue, see [SECURITY.md](SECURITY.md).

Issues labelled
[`good first issue`](https://github.com/growingnet/gromo/labels/good%20first%20issue)
are a good place to start. For larger changes, open an issue or a discussion first,
so that we can agree on the approach before you write the code.

## Development setup

1. Fork the repository on GitHub, clone your fork, replacing `$USER` with your
GitHub username and add the main repository as `upstream`:

   ```bash
    git clone git@github.com:$USER/gromo.git --origin origin
    cd gromo
    git remote add upstream git@github.com:growingnet/gromo.git
    git remote set-url --push upstream no-pushing
    git fetch upstream
    git branch --set-upstream-to=upstream/main main
    ```

2. Install the package with its development dependencies using
   [uv](https://docs.astral.sh/uv/), and the pre-commit hooks:

   ```bash
   uv sync --all-extras
   uv run pre-commit install
   ```

   Without uv: `pip install -e ".[dev,test,doc]"`, then `pre-commit install`.

## Making changes

1. Create a branch from an up-to-date `main`:

   ```bash
   git fetch upstream
   git switch -c <branch-name> upstream/main
   ```

2. Make your changes, with tests and documentation:

   - Code is formatted and linted with [ruff](https://docs.astral.sh/ruff/), run by
     the pre-commit hooks at each commit.
   - Types are checked with [ty](https://docs.astral.sh/ty/), adopted gradually: the
     existing code still has type errors, so the CI only reports ty diagnostics on
     the lines changed by a pull request, without failing. New or modified code
     should have none. Run ty on the package with
     `uv run pre-commit run ty --hook-stage manual --all-files`, or use the ty
     extension of your editor.
   - Public functions and classes have type annotations and
     [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings,
     checked by pydoclint.
   - Tests use `unittest` and live in `tests/`:

     ```bash
     uv run python -m unittest                                # all tests
     uv run python -m unittest tests.test_growing_container   # one module
     ```

   - The documentation is built with Sphinx: `uv run make -C docs html`, then open
     `docs/build/html/index.html`.

3. Add an entry to [docs/source/whats_new.rst](docs/source/whats_new.rst), in the
   "Develop branch" section, under "Enhancements", "Bugs" or "API changes" (for
   backward-incompatible changes):

   ```rst
   - Short description of the change (:gh:`123` by `Your Name`_)
   ```

   where `123` is the number of your pull request, and your name is defined once at
   the end of the file: `.. _Your Name: https://github.com/<your-username>`. A CI
   check requires this entry; maintainers can add the `no changelog` label to pull
   requests that do not need one.

4. Run all the checks before pushing: `uv run pre-commit run --all-files`.

## Pull requests

- Push your branch to your fork and open a pull request to `growingnet/gromo:main`.
  Draft pull requests are welcome to get early feedback.
- Pull requests are squash-merged: their title becomes the commit message on `main`.
  Write it in the [Conventional Commits](https://www.conventionalcommits.org/)
  format, for example `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`,
  `refactor: ...`, `build: ...` or `ci: ...`.
- The CI runs the tests on Python 3.11 to 3.14, the linters and the documentation
  build. A pull request needs the approval of a maintainer to be merged.
- You remain responsible for the code you contribute, whatever tools you use to
  write it: make sure that you understand it, that it works, and that you have the
  right to contribute it.

## Licence and authorship

- Contributions are accepted under the [BSD-3-Clause licence](LICENCE.md) of the
  project.
- Contributors with at least one pull request merged in a release are added as
  authors of this release, in `CITATION.cff` (also used for the Zenodo archive) and
  `pyproject.toml`. Other contributions can be credited by collective decision.
  To be listed with your ORCID and affiliation, add them to `CITATION.cff` in your
  pull request, or send them to a maintainer.

## Maintainers

The maintainers review and merge pull requests, and prepare the releases.
Reports of Code of Conduct violations go to the project supervisors, see
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md#reporting-an-issue).

Maintainers:

- Sylvain Chevallier ([@sylvchev](https://github.com/sylvchev))
- Guillaume Charpiat
