# Release

MLX-Gen publishes Python distributions to PyPI and attaches the same distributions to a GitHub Release through `.github/workflows/release.yml`.

## Required GitHub And PyPI Setup

PyPI trusted publishing must be configured for:

- PyPI project: `mlx-gen-jzg`
- GitHub repository: `jiangzhenguo/mlx-gen`
- Workflow filename: `release.yml`
- GitHub environment: `pypi`

The workflow uses OpenID Connect through `pypa/gh-action-pypi-publish`; it does not require a long-lived PyPI API token.

The workflow uses the `pypi` environment for trusted publishing, then uses the `github-release`
environment only after PyPI publication succeeds. Configure environment protection rules in GitHub
if releases should require manual approval.

## Release Checks

The release workflow:

- runs fast tests on macOS;
- validates package name, version, changelog entry, tag name, and duplicate PyPI versions;
- builds distributions with `uv build`;
- validates distributions with `twine check`;
- uploads build artifacts;
- publishes the built artifacts to PyPI through trusted publishing;
- creates a GitHub Release with the same wheel and source distribution only after PyPI succeeds.

## Rehearsal

Run a non-publishing rehearsal from GitHub Actions:

```sh
gh workflow run release.yml \
  -f version=<version> \
  -f publish=false
```

This validates the release metadata and builds artifacts, but does not create a GitHub Release or publish to PyPI.

## Publish

The preferred release path is a version tag:

```sh
git tag v<version>
git push origin main --tags
```

Publishing can also be triggered manually when the target commit has already been pushed:

```sh
gh workflow run release.yml \
  -f version=<version> \
  -f publish=true \
  -f publish_confirmation=publish-mlx-gen-<version>
```

For manual publishing, the confirmation string must match `publish-mlx-gen-<version>`.
