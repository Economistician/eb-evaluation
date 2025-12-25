# Electric Barometer – Release Procedure

This document defines the **canonical release process** for all Electric Barometer
repositories. It is written in Markdown syntax but intentionally stored as a `.txt`
file for portability and explicitness.

---

## Supported Repositories

This process applies to:

- `eb-metrics`
- `eb-evaluation`
- `eb-adapters`
- `eb-features`
- `electric-barometer` (umbrella package)

All repositories follow the **same release spine** with only minor substitutions
(package name, version, smoke import).

---

## Release Preconditions

Before starting a release, verify:

- `git status` is clean
- All tests pass locally
- `pyproject.toml` version is correct and final
- Dependency bands are already aligned
- You are on `main`
- You have a valid PyPI API token

---

## Step 1 – Version Confirmation

Confirm the version declared in `pyproject.toml`:

```toml
[project]
version = "X.Y.Z"
```

The version **must not already exist** on PyPI.

Check existing versions:

```bash
python -m pip index versions <package-name>
```

---

## Step 2 – Clean Build Artifacts

Always start from a clean state:

```bash
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
```

---

## Step 3 – Build Distributions

Build both sdist and wheel:

```bash
python -m build
```

Expected output:
- `dist/<package>-X.Y.Z.tar.gz`
- `dist/<package>-X.Y.Z-py3-none-any.whl`

---

## Step 4 – Validate Distributions

Before uploading, validate artifacts:

```bash
python -m twine check dist/*
```

This **must pass** with no errors.

---

## Step 5 – Upload to PyPI

Upload artifacts to PyPI:

```bash
python -m twine upload dist/*
```

Notes:
- PyPI **does not allow file overwrites**
- If upload fails due to existing files, **bump the version**
- Ignore trusted publishing warnings unless configured

---

## Step 6 – Tag the Release

Create and push a Git tag matching the version:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Tags must always correspond to **published artifacts**.

---

## Step 7 – Install Verification

Force-install the released version directly from PyPI:

```bash
python -m pip install --no-cache-dir -U -i https://pypi.org/simple <package-name>==X.Y.Z
```

Confirm installation:

```bash
python -m pip show <package-name>
```

---

## Step 8 – Smoke Import Test

Run a minimal import test:

```bash
python -c "import <package>; print('import OK')"
```

For umbrella release validation:

```bash
python -c "import eb_metrics, eb_evaluation, eb_adapters, eb_features, electric_barometer; print('EB smoke OK')"
```

---

## Step 9 – Dependency Integrity Check

Verify the environment has no conflicts:

```bash
python -m pip check
```

Expected output:

```
No broken requirements found.
```

---

## Versioning Rules

- Patch releases (`X.Y.Z+1`) for:
  - Dependency band changes
  - Metadata fixes
  - Packaging corrections

- Minor releases (`X.Y+1.0`) for:
  - New features
  - New public APIs

- Major releases (`X+1.0.0`) for:
  - Breaking API changes

---

## Non-Negotiable Rules

- Never reuse a PyPI version
- Never retag an existing version
- Never publish without testing install
- Never publish umbrella until all leaves are live

---

## Final Checklist

- [ ] Clean git state
- [ ] Build succeeds
- [ ] Twine check passes
- [ ] PyPI upload succeeds
- [ ] Git tag pushed
- [ ] pip install verified
- [ ] Smoke import passes
- [ ] pip check clean

---

**This document is the single source of truth for Electric Barometer releases.**