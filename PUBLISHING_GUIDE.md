# PyPI Publishing Guide for POLARIS

This guide explains how to set up and use automated PyPI publishing for the POLARIS package.

## 🏗️ GitHub Actions Workflows

Three workflows have been created:

### 1. `ci.yml` - Continuous Integration
- **Triggers**: Push/PR to main branches
- **Purpose**: Run tests, code quality checks, build testing
- **Platforms**: Ubuntu, Windows, macOS
- **Python versions**: 3.8, 3.9, 3.10, 3.11

### 2. `test-publish.yml` - Test PyPI Publishing
- **Triggers**: Tags like `v2.0.0-test`, `v2.0.0-alpha`, `v2.0.0-beta`
- **Purpose**: Publish to Test PyPI for testing
- **Repository**: https://test.pypi.org/

### 3. `publish.yml` - Production PyPI Publishing
- **Triggers**: GitHub releases
- **Purpose**: Publish to production PyPI
- **Repository**: https://pypi.org/

## 🔐 Setting Up Trusted Publishing

PyPI now supports "Trusted Publishing" which eliminates the need for API tokens. Here's how to set it up:

### Step 1: Configure PyPI Trusted Publishing

1. Go to [PyPI](https://pypi.org/) and log in
2. Go to your project page or create a new project placeholder
3. Navigate to "Settings" → "Publishing"
4. Click "Add a new publisher"
5. Fill in the details:
   - **PyPI Project Name**: `polaris-marl`
   - **Owner**: `ecdogaroglu` (your GitHub username)
   - **Repository name**: `polaris`
   - **Workflow filename**: `publish.yml`
   - **Environment name**: `pypi`

### Step 2: Configure Test PyPI (Optional)

Repeat the same process on [Test PyPI](https://test.pypi.org/) with:
- **Environment name**: `testpypi`
- **Workflow filename**: `test-publish.yml`

## 📦 Publishing Process

### Option 1: Production Release (Recommended)

1. **Update version** in `polaris/__init__.py`:
   ```python
   __version__ = "2.1.0"  # Update this
   ```

2. **Commit and push** your changes:
   ```bash
   git add .
   git commit -m "Release v2.1.0"
   git push origin main
   ```

3. **Create a GitHub Release**:
   - Go to your GitHub repository
   - Click "Releases" → "Create a new release"
   - Choose "Create new tag" and enter `v2.1.0`
   - Fill in release title and description
   - Click "Publish release"

4. **Automatic publishing**: The workflow will automatically:
   - Run tests on multiple Python versions
   - Build the package
   - Publish to PyPI

### Option 2: Test Publishing

For testing before production release:

1. **Create a test tag**:
   ```bash
   git tag v2.1.0-test
   git push origin v2.1.0-test
   ```

2. **Check Test PyPI**: Your package will be published to https://test.pypi.org/project/polaris-marl/

3. **Test installation**:
   ```bash
   pip install -i https://test.pypi.org/simple/ polaris-marl
   ```

### Option 3: Manual Publishing

Using the build script:

```bash
# Test publishing
python scripts/build_and_publish.py --test-pypi

# Production publishing (requires confirmation)
python scripts/build_and_publish.py --prod-pypi
```

## 🔧 GitHub Repository Settings

### Required Settings

1. **Enable GitHub Actions**: Settings → Actions → General → "Allow all actions"

2. **Create Environments** (Settings → Environments):
   - **Environment name**: `pypi`
     - **Deployment protection rules**: Optionally add reviewers
     - **Environment secrets**: None needed (using trusted publishing)
   
   - **Environment name**: `testpypi` (optional)
     - Same settings as above

3. **Branch Protection** (optional but recommended):
   - Protect `main` branch
   - Require status checks from CI workflow
   - Require pull request reviews

## 📋 Workflow Details

### CI Workflow Features
- **Cross-platform testing**: Ubuntu, Windows, macOS
- **Multiple Python versions**: 3.8-3.11
- **Code quality checks**: black, isort, flake8
- **GNN dependency testing**: Separate job for torch-geometric
- **Package installation testing**: Build and install wheel

### Publishing Workflow Features
- **Automatic versioning**: Reads from `polaris/__init__.py`
- **Security**: Uses OpenID Connect, no API tokens needed
- **Quality gates**: Only publishes if tests pass
- **Artifact management**: Builds are uploaded and downloaded between jobs

## 🚀 Release Checklist

Before creating a release:

- [ ] Update version in `polaris/__init__.py`
- [ ] Update `CHANGELOG.md` (if you have one)
- [ ] Run tests locally: `pytest tests/`
- [ ] Run code quality checks: `black polaris/ && isort polaris/ && flake8 polaris/`
- [ ] Test build locally: `python -m build`
- [ ] Consider testing with Test PyPI first
- [ ] Create GitHub release with proper release notes

## 🐛 Troubleshooting

### Common Issues

1. **"Trusted publishing not configured"**
   - Ensure you've set up trusted publishing on PyPI
   - Check that the environment name matches exactly
   - Verify the repository and workflow filename are correct

2. **"Tests failing"**
   - Check the Actions tab for detailed error logs
   - Ensure all dependencies are properly specified
   - Test locally first

3. **"Package already exists"**
   - Version numbers must be unique
   - Increment the version in `polaris/__init__.py`
   - You cannot overwrite existing versions on PyPI

4. **"Workflow not triggering"**
   - Ensure the workflow file is on the main branch
   - Check the trigger conditions (tags, releases)
   - Verify GitHub Actions are enabled

### Getting Help

- **GitHub Actions logs**: Check the "Actions" tab in your repository
- **PyPI status**: Check https://status.python.org/
- **Package validation**: Use `twine check dist/*` locally

## 📈 Monitoring

After publishing:

- **PyPI project page**: https://pypi.org/project/polaris-marl/
- **Download statistics**: Available on the PyPI project page
- **GitHub repository insights**: Traffic, clones, etc.

## 🔄 Updating the Workflows

To modify the workflows:

1. Edit the `.github/workflows/*.yml` files
2. Test changes on a fork or feature branch first
3. Workflows are automatically updated when you push to main

Remember: Workflow changes only affect future runs, not the current one.

---

**Note**: The first release might require manual intervention to create the PyPI project. After that, all releases can be fully automated. 