# Releasing regime-shap

Steps to cut a release, mint a Zenodo DOI, and publish to PyPI. These are gated on external
accounts and are done deliberately, not automatically. Do them only when the package is ready to
be a real release.

## 0. Pre-release checks

- CI is green on `develop` and `main`, and the docs build cleanly.
- `pyproject.toml` and `regime_shap/__init__.py` declare the same version.
- In `CHANGELOG.md`, move the `[Unreleased]` entries under a new `[0.1.0]` heading with the date.
- In `CHANGELOG.md`, remove the "Nothing has been released yet" line now that a version exists.
- In `README.md`, replace the "Not yet on PyPI, install from source" Installation block with `pip install regime-shap` as the primary method (keep the source install for contributors), and remove the remaining pre-release wording.
- In `space/README.md`, update the "pre-release and not yet on PyPI" line to reflect the published release.

## 1. Connect Zenodo (one time)

- Sign in at https://zenodo.org with your GitHub account.
- Go to your account settings, then GitHub, and switch on `faithcodes-lab/regime-shap`.
- From then on, Zenodo archives each new GitHub release and mints a DOI for it, plus a concept DOI
  that always points to the latest version.

## 2. Configure PyPI publishing (one time)

The `publish.yml` workflow is set up for PyPI Trusted Publishing (OIDC), which needs no stored
token. On https://pypi.org, add a Trusted Publisher for the project with:

- Owner: `faithcodes-lab`
- Repository: `regime-shap`
- Workflow: `publish.yml`
- Environment: `pypi`

Alternatively, add a `PYPI_API_TOKEN` repository secret and switch `publish.yml` to token auth
(the token lines are already there, commented out).

## 2b. Rehearse on TestPyPI (recommended)

TestPyPI is a throwaway sandbox. Rehearsing there first confirms the publish mechanism works and
lets you see the rendered project page before the real, irreversible upload. The
`publish-testpypi.yml` workflow does this on demand, and gives each run a unique dev version so it
can be repeated.

One-time setup:

- Sign in at https://test.pypi.org with your GitHub account and turn on 2FA.
- Add a pending publisher on TestPyPI (Publishing, then Add a pending publisher):
  - Project name: `regime-shap`
  - Owner: `faithcodes-lab`
  - Repository name: `regime-shap`
  - Workflow name: `publish-testpypi.yml`
  - Environment name: `testpypi`
- In the GitHub repo Settings, under Environments, create one named `testpypi`.

To rehearse:

- In the repo, open the Actions tab, choose "Rehearse publish (TestPyPI)", and Run workflow.
- When it finishes, check the page at https://test.pypi.org/project/regime-shap/.

## 3. Cut the release

- Promote `develop` to `main` via a pull request, so `main` holds the release state.
- Create a GitHub release from `main` with the tag `v0.1.0`.
  - This triggers `publish.yml`, which builds the package and publishes it to PyPI.
  - Zenodo archives the release and mints the DOI.

## 4. After the DOI exists

- Add the DOI to `CITATION.cff` (a `doi:` field) and set `date-released` to the release date.
- Add a Zenodo DOI badge and a PyPI badge to `README.md`, which are now truthful.
- Commit these on a small follow-up pull request.
