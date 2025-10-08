# Contributing to `fitsmap`

Thank you for helping to make `fitsmap` work better for you and for the larger
community! This document outlines some steps that will hopefully make
contributing to the project as simple and painless as possible.


## Working with the code

### Getting the code

To contribute to the repository, you need to work with a "fork" of the code. To
fork the code into a space that you own, follow this link
[fork](https://github.com/ryanhausen/fitsmap/fork) or click the fork button on
the main repository page. Once the fork is created, you can clone the forked
version of the repository and start making your changes!

```bash
git clone https://github.com/<username or orgname>/fitsmap
```

Next, add the main repository as an upstream remote, so that you can merge any
updates to the main repository that happen while you are making your changes.

```bash
git remote add upstream https://github.com/ryanhausen/fitsmap
```

Next, you'll make sure everything is up to date by fetching and then making a
new branch based on the current state of the main repository.

```bash
git fetch --all
git checkout upstream/master
git checkout -b your-feature-name
```

Here `your-feature-name` should be a descriptive name for the branch you're
working on. It could be something like `issue-39` or `faster-tiling-algo`. You
are now ready to start making changes!

### Making changes

This repository uses [`uv`](https://docs.astral.sh/uv/) for environment
management and has a Makefile for running the tasks for formatting, style, and
testing.

#### Updating the environment

If you need to add or remove a dependency, you have to do it through `uv`. If
you don't, then the CI/CD checks may not work. You can use [`uv
add`](https://docs.astral.sh/uv/concepts/projects/dependencies/#adding-dependencies)
to add dependencies and [`uv
remove`](https://docs.astral.sh/uv/concepts/projects/dependencies/#removing-dependencies)
to remove dependencies.  See the [`uv`](https://docs.astral.sh/uv/)
documentation for more details and other commands.

#### Updating the code

Before you submit your pull request, and perhaps periodically while you develop,
you should check to see if your code conforms to the formatting, style,
security, and testing requirements, and does not break the documentation build.
The Makefile included in the repo can run these checks for you:

- Building docs: `make docs` -- Runs sphinx to build the website and API
documentation that is hosted on
[readthedocs](https://fitsmap.readthedocs.io/en/latest/). If you add any new
functions, please add type hints and docstrings. They are used by sphinx to
generate the API documentation automatically.

- Security: `make check-security` -- Runs the tool
[bandit](https://bandit.readthedocs.io/en/latest/) to look for security
vulnerabilities in the code.

- Formatting: `make format` -- Runs the `ruff` implementation of the `black`
style guide and import sorting rules.

- Style: `make check-style` -- Runs [`ruff`](https://astral.sh/ruff) linter to
check linting and style rules.

- Testing: `make test` -- Runs `pytest` to check the unit and integration tests
and report the coverage. If you add new functionality, please add tests for your
code. One of the checks that gets run on a pull request is to check for code
test coverage, so keeping this in mind and adding tests as you go will lessen
the burden of writing tests. If you are not sure where the current test coverage
stands, after you run `make test`, run the following:

```bash
cd htmlcov
python -m http.server
```

Then open the URL in the terminal, and you can interactively see how much of
your code is covered by tests.

> [!IMPORTANT]
> If you make changes to a part of the code that affects the contents of output
> files (`.png`, `.cbor`, `.html`, `.js`), then the integration tests will fail
> because they compare the file content from the current release with your
> changes. If you run into this issue, submit your pull request when ready, and
> I will help update the files that need to be updated so that your code passes
> all of the tests.

## License

By contributing, you agree that your contributions will be licensed under the
same open-source license as the project.

## Authorship

Thanks for contributing! If you would like to be added to the list of authors,
please add yourself to the list of authors in the `pyproject.toml` file.

-----
This document is based on the JHU-SSEC base template contributing
[document](https://github.com/ssec-jhu/base-template/blob/main/CONTRIBUTING.md)
