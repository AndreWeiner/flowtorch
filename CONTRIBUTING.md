# How to contribute to *flowTorch*

We appreciate all efforts contributing to the *flowTorch* project, may it be bug-fixes, feature contributions, feature suggestions, additional examples, or other kinds of improvements. If you would like to contribute, you may consider the following steps.

## 0. Open a new issue

It is always useful to open a new issue as a first step. The issue helps the developers to plan and organize developments and to provide quick feedback on potential problems or existing solutions. For example, it might be that the bug you are reporting has already been fixed on a development branch or that someone is already working on a similar feature to the one you are suggesting. *flowTorch* is still a rather small project, so there are typically few open issues. Nonetheless, you should give your issue a suitable label to follow common best practices (e.g., feature, bug, documentation, ...).

## 1. Fork the *flowTorch* repository and create a new branch

The typical workflow of forking and branching is very well described in the [GitHub documentation](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

The `develop` branch is the integration branch for ongoing development. Create
feature, bug-fix, and documentation branches from `develop`, and target
`develop` with the corresponding pull requests. The `main` branch represents
released versions and normally receives release merges from `develop`. A
critical hotfix may instead start from and target `main`; merge the hotfix back
into `develop` afterwards.

## 2. Ensure code quality

*flowTorch* uses the [PyTorch library](https://pytorch.org/docs/stable/index.html) as backend for array-like data structures (tensors) and operations thereon. When implementing new features, try to rely as much as possible on the functionality offered by PyTorch instead of using NumPy, SciPy or similar libraries.

Most of the library contains [type hints](https://docs.python.org/3/library/typing.html). Type hints are not strictly necessary to run the code, but they make the lives of everybody much easier, so please use type hint in all parts of your code.

Python is a language that allows implementing operations with enormous complexity in a single line of code. Therefore, it is extremely important to provide a detailed documentation of new functionality containing all considerations the developer had in mind and also potential references or resources that were used as basis. *flowTorch* generates the documentation using [Sphinx](https://www.sphinx-doc.org/en/master/), and therefore, doc-stings should be formatted as [reStructuredText](https://docutils.sourceforge.io/rst.html).

Python code is formatted with [Black](https://black.readthedocs.io/) using a
line length of 88 characters. Run the formatter through tox before submitting
your changes:

```bash
tox -e format
```

Check that formatting is correct without modifying any files with:

```bash
tox -e format-check
```

Python code is linted with [Ruff](https://docs.astral.sh/ruff/). Run the same
lint check that is used by continuous integration with:

```bash
tox -e lint
```

These commands check the `flowtorch` package and the test suite by default.
Specific files or directories can be supplied after `--`, for example:

```bash
tox -e lint -- flowtorch/analysis tests/analysis
```

Type annotations can optionally be checked with:

```bash
tox -e type-check
```

## 3. Provide unit tests

If new features are added, accompanying unit tests should be provided. We use [PyTest](https://docs.pytest.org/en/6.2.x/) for testing. If the tests require additional datasets, please make sure that you have the permission to share the data such that the new data can be added to the *flowTorch* datasets in the next release. It might be necessary in some cases to create fake data (data that behave the same way as real data but that might be smaller and not protected).

## 4. Push changes and create a pull-request

To complete your contribution, push your branch and create a new [pull
request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
against the `develop` branch of the *flowTorch* repository. Automated checks
will verify formatting, linting, and the dataset-free test suite. The
type-checking job is currently optional and does not block merging. If datasets
are required, please provide a download link so that the integration tests or
examples can be executed.

## Example contribution workflow

The following commands illustrate how to contribute a feature named
`improve-svd-api`. Replace `<your-github-username>` and the example branch name
with your own values.

First, fork the repository on GitHub. Then clone your fork and add the upstream
*flowTorch* repository as a remote:

```bash
git clone https://github.com/<your-github-username>/flowtorch.git
cd flowtorch
git remote add upstream https://github.com/AndreWeiner/flowtorch.git
```

Update your local `develop` branch and create a feature branch from it:

```bash
git switch develop
git pull --ff-only upstream develop
git switch -c feature/improve-svd-api
```

Implement the feature and its tests. Then run the local quality checks and
dataset-free tests:

```bash
tox -e format
tox -e format-check
tox -e lint
tox -e py310
```

Type checking is optional:

```bash
tox -e type-check
```

Review and commit the changes:

```bash
git status
git diff
git add flowtorch tests
git commit -m "Add improved SVD API"
```

Push the feature branch to your fork:

```bash
git push -u origin feature/improve-svd-api
```

Finally, create a pull request targeting the upstream `develop` branch. This can
be done in the GitHub web interface or with the GitHub CLI:

```bash
gh pr create \
  --repo AndreWeiner/flowtorch \
  --base develop \
  --head <your-github-username>:feature/improve-svd-api \
  --title "Add improved SVD API" \
  --body "Implements the proposed SVD API improvement and adds tests."
```

**Thank you for considering to contribute to the *flowTorch* project!**
