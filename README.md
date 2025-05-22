# python-lucide

A Python package for working with [Lucide icons](https://lucide.dev/). This package allows you to:

1. Build a SQLite database of Lucide icons
2. Serve Lucide SVG icons from a Python application
3. Customize icons with classes and attributes

## Installation

### Basic Installation
```bash
pip install lucide
```

This installs the core package without the pre-built icons database. You'll need to build the database yourself using the `lucide-db` command.

### Installation with Pre-built Database
```bash
pip install "lucide"
```

This installs the package with a pre-built Lucide icons database, so you can use it right away without building it yourself.

### Development Installation
If you plan to contribute to `python-lucide`, see the "Development" section for setup instructions.

## Usage

### Getting an Icon

```python
from lucide import lucide_icon

# Get a basic icon
svg_content = lucide_icon("home")

# Add a CSS class
svg_content = lucide_icon("settings", cls="my-icon-class")

# Add custom attributes
svg_content = lucide_icon("arrow-up", attrs={"width": "32", "height": "32"})

# Provide fallback text for when an icon is missing
svg_content = lucide_icon("some-icon", fallback_text="Icon")
```

### Building the Icon Database

#### Using the Command-Line Tool

The package installs a command-line tool called `lucide-db` that you can use to build the database:

```bash
# Build a database with all icons
lucide-db

# Specify a custom output path
lucide-db -o /path/to/output.db

# Use a specific Lucide version
lucide-db -t 0.500.0

# Include only specific icons
lucide-db -i home,settings,user

# Include icons from a file (one name per line)
lucide-db -f my-icons.txt

# Enable verbose output
lucide-db -v
```

#### Using the Python API

You can also build the database programmatically:

```python
from lucide.cli import download_and_build_db

# Build a custom database
db_path = download_and_build_db(
    output_path="custom-icons.db",
    tag="0.511.0",
    icon_list=["home", "settings", "user"]
)
```

### Getting a List of Available Icons

```python
from lucide import get_icon_list

# Get all available icon names
icons = get_icon_list()
print(icons)  # ['activity', 'airplay', 'alert-circle', ...]
```

## Configuration

The package will look for the icons database in the following locations (in order):

1. The path specified in the `LUCIDE_DB_PATH` environment variable
2. In the package data directory (if installed with the `db` extra)
3. In the current working directory as `lucide-icons.db`

## Example Web Framework Integration

### Flask

```python
from flask import Flask
from lucide import lucide_icon

app = Flask(__name__)

@app.route('/icons/<icon_name>')
def serve_icon(icon_name):
    svg = lucide_icon(icon_name, cls="my-icon")
    return svg, 200, {'Content-Type': 'image/svg+xml'}
```

### FastAPI

```python
from fastapi import FastAPI
from fastapi.responses import Response
from lucide import lucide_icon

app = FastAPI()

@app.get("/icons/{icon_name}")
def serve_icon(icon_name: str):
    svg = lucide_icon(icon_name, cls="my-icon")
    return Response(content=svg, media_type="image/svg+xml")
```

## Development

This project uses `uv` for project and virtual environment management, and `pre-commit` for code quality checks. A `Makefile` is also provided for common development tasks.

### Setup Development Environment

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mmacpherson/python-lucide.git
    cd python-lucide
    ```

2.  **Create a virtual environment and install dependencies:**
    This project uses `uv` for fast environment management and dependency installation.
    ```bash
    # Create a virtual environment in .venv/
    uv venv
    # Activate the virtual environment
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows (PowerShell):
    # .venv\\Scripts\\Activate.ps1
    # On Windows (CMD):
    # .venv\\Scripts\\activate.bat

    # Install the package in editable mode with development dependencies
    uv pip install -e ".[dev]"
    ```
    This command installs `lucide` in "editable" mode (`-e`), meaning changes you make to the source code will be reflected immediately. It also installs all dependencies listed under the `[dev]` extra in your `pyproject.toml` (like `pytest`, `ruff`, and `pre-commit`).

    Alternatively, you can use the `Makefile` target:
    ```bash
    make env
    # Then activate the environment as shown above.
    ```

3.  **Install pre-commit hooks:**
    ```bash
    uv run pre-commit install
    ```
    Or use the Makefile:
    ```bash
    make install-hooks
    ```

<details>
<summary>Alternative: Using Python's venv and pip</summary>

If you prefer not to use `uv` or `make`, you can use Python's built-in `venv` module and `pip`:

1.  **Clone the repository (if not already done):**
    ```bash
    git clone https://github.com/mmacpherson/python-lucide.git
    cd python-lucide
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows (PowerShell):
    # .venv\\Scripts\\Activate.ps1
    # On Windows (CMD):
    # .venv\\Scripts\\activate.bat
    ```

3.  **Install dependencies:**
    ```bash
    pip install -e ".[dev]"
    ```

4.  **Install pre-commit hooks:**
    ```bash
    pre-commit install
    ```

</details>

### Running Tests

After setting up your development environment, run tests using `pytest`.
You can use the `Makefile`:
```bash
make test
```
Or run `pytest` directly via `uv`:
```bash
uv run pytest
```

### Linting and Formatting

This project uses `ruff` for linting and formatting, managed via `pre-commit`. The hooks will run automatically on commit.
To run all hooks manually across all files, use the `Makefile`:
```bash
make run-hooks-all-files
```
Or run `pre-commit` directly via `uv`:
```bash
uv run pre-commit run --all-files
```

You can also run `ruff` commands directly:
```bash
uv run ruff check .
uv run ruff format .
```

### Building the Icon Database

The packaged database can be rebuilt using the `lucide-db` command-line tool. To update the database bundled with the `[db]` extra (typically stored at `src/lucide/data/lucide-icons.db`):

Using the `Makefile` (recommended for consistency):
```bash
# Replace <version> with the desired Lucide tag, e.g., 0.511.0
make db TAG=<version>
```
Or directly using `lucide-db` (ensure your virtual environment is active):
```bash
# Replace <version> with the desired Lucide tag
uv run lucide-db -o src/lucide/data/lucide-icons.db -t <version> -v
```
The default tag used by `make db` is specified in the `Makefile`.

### Building the Package

To build the sdist and wheel for distribution:
```bash
python -m build
```
Or, if you have `hatch` installed (it's part of `[dev]` dependencies):
```bash
hatch build
```

### Makefile Targets
A `Makefile` is provided with common development tasks. Run `make help` to see available targets, including:
*   `env`: Sets up the development environment (creates `.venv` and installs dependencies).
*   `db`: Rebuilds the Lucide icon database.
*   `test`: Runs tests.
*   `install-hooks`: Installs pre-commit hooks.
*   `run-hooks-all-files`: Runs all pre-commit hooks on all files.
*   `clean`: Removes build artifacts, `__pycache__`, etc.
*   `nuke`: A more thorough clean, including the `uv` cache if present and the `.venv` directory.


## License

This project is licensed under the MIT License - see the LICENSE file for details.
