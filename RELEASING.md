# Releasing regime-shap

Steps to cut a release, mint a Zenodo DOI, and publish to PyPI. These are gated on external
accounts and are done deliberately, not automatically. Do them only when the package is ready to
be a real release.

## 0. Pre-release checks

- CI is green on `develop` and `main`, and the docs build cleanly.
- `pyproject.toml` and `regime_shap/__init__.py` declare the same version.
- In `CHANGELOG.md`, move the `[Unreleased]` entries under a new `[0.1.0]` heading with the date.
- Update `README.md` and `CHANGELOG.md` wording from pre-release to the released version.

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

## 3. Cut the release

- Promote `develop` to `main` via a pull request, so `main` holds the release state.
- Create a GitHub release from `main` with the tag `v0.1.0`.
  - This triggers `publish.yml`, which builds the package and publishes it to PyPI.
  - Zenodo archives the release and mints the DOI.

## 4. After the DOI exists

- Add the DOI to `CITATION.cff` (a `doi:` field) and set `date-released` to the release date.
- Add a Zenodo DOI badge and a PyPI badge to `README.md`, which are now truthful.
- Commit these on a small follow-up pull request.
