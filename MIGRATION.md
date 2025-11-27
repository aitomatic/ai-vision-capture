# Migration Guide: Poetry to UV

This document provides instructions for migrating from Poetry to UV for package management and building.

## What Changed?

The project has been migrated from Poetry to UV for faster dependency management and builds. UV is a modern, blazing-fast Python package installer and resolver written in Rust.

### Key Changes

1. **Build Backend**: Changed from `poetry-core` to `hatchling`
2. **Package Metadata**: Migrated from Poetry-specific format to PEP 621 standard
3. **Lock File**: `poetry.lock` → `uv.lock`
4. **Commands**: Poetry commands replaced with UV equivalents
5. **CI/CD**: GitHub Actions workflows updated to use UV

## For Contributors

### Installing UV

If you're a contributor or working on local development, you'll need to install UV:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip (if you prefer)
pip install uv
```

### Migration Steps

1. **Remove Poetry artifacts**:
   ```bash
   rm poetry.lock
   rm -rf .venv
   ```

2. **Install dependencies with UV**:
   ```bash
   uv sync --all-extras
   ```

3. **Verify installation**:
   ```bash
   uv run pytest
   ```

## Command Comparison

| Task | Poetry Command | UV Command |
|------|---------------|------------|
| Install dependencies | `poetry install` | `uv sync` |
| Install with extras | `poetry install --with dev` | `uv sync --all-extras` |
| Add dependency | `poetry add package` | `uv add package` |
| Add dev dependency | `poetry add --group dev package` | `uv add --dev package` |
| Remove dependency | `poetry remove package` | `uv remove package` |
| Run command | `poetry run pytest` | `uv run pytest` |
| Build package | `poetry build` | `uv build` |
| Publish to PyPI | `poetry publish` | `uv publish` |
| Update dependencies | `poetry update` | `uv lock --upgrade` |
| Show dependencies | `poetry show` | `uv tree` |

## Using the Makefile

The project Makefile has been updated to use UV. All commands remain the same:

```bash
make setup      # Install dependencies
make test       # Run tests
make lint       # Run linters
make format     # Format code
make build      # Build package
make publish    # Publish to PyPI
```

## CI/CD Changes

The GitHub Actions workflows have been updated to use UV:

- **`.github/workflows/ci.yml`**: Uses `astral-sh/setup-uv@v4` action
- **`.github/workflows/publish.yml`**: Uses UV for building and publishing

No changes needed to your workflow - just push your changes and the CI will run with UV.

## Benefits of UV

- **Speed**: 10-100x faster than pip and Poetry
- **Reliability**: Better dependency resolution
- **Compatibility**: Works with standard Python packaging (PEP 621)
- **Simplicity**: Single tool for package management and building
- **Modern**: Written in Rust, actively maintained by Astral

## Troubleshooting

### Issue: `uv` command not found

**Solution**: Make sure UV is installed and in your PATH:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc
```

### Issue: Dependencies not resolving

**Solution**: Try clearing the cache and re-syncing:
```bash
uv cache clean
uv sync --all-extras
```

### Issue: Pre-commit hooks failing

**Solution**: Re-install pre-commit hooks:
```bash
uv run pre-commit install
```

## FAQ

**Q: Can I still use Poetry?**
A: While the project configuration has been migrated to standard PEP 621 format, Poetry should still work for most operations. However, we recommend using UV for better performance and to stay aligned with the project's development workflow.

**Q: Do I need to uninstall Poetry?**
A: No, Poetry and UV can coexist. You can keep Poetry installed if you use it for other projects.

**Q: Will this break my existing development environment?**
A: You'll need to run `uv sync` to create a new virtual environment, but your code and configuration won't be affected.

**Q: How do I specify a Python version?**
A: UV respects the `requires-python` field in `pyproject.toml`. You can also use:
```bash
uv python install 3.10
uv venv --python 3.10
```

## Additional Resources

- [UV Documentation](https://docs.astral.sh/uv/)
- [UV GitHub Repository](https://github.com/astral-sh/uv)
- [PEP 621 - Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)

## Need Help?

If you encounter any issues during migration, please:
1. Check the [troubleshooting section](#troubleshooting) above
2. Review the [UV documentation](https://docs.astral.sh/uv/)
3. Open an issue on the repository with details about your problem
