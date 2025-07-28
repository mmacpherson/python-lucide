# python-lucide

[![PyPI version](https://badge.fury.io/py/python-lucide.svg)](https://badge.fury.io/py/python-lucide)
[![Python versions](https://img.shields.io/pypi/pyversions/python-lucide.svg)](https://pypi.org/project/python-lucide/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/mmacpherson/python-lucide/workflows/CI/badge.svg)](https://github.com/mmacpherson/python-lucide/actions/workflows/ci.yml)
[![Lucide Version](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmmacpherson%2Fpython-lucide%2Fmain%2F.github%2Flucide-version.json)](https://github.com/lucide-icons/lucide/releases)
[![PyPI downloads](https://img.shields.io/pypi/dm/python-lucide.svg)](https://pypi.org/project/python-lucide/)
[![Built with uv](https://img.shields.io/badge/Built%20with-uv-purple.svg)](https://github.com/astral-sh/uv)

A Python package that provides easy access to all [Lucide
icons](https://lucide.dev/) as SVG strings. Just import and use any Lucide icon
in your Python projects, with no javascript in sight.

## Features
- 🎨 **Access 1600+ Lucide icons** directly from Python
- 🛠 **Customize icons** with classes, sizes, colors, and other SVG attributes
- 🚀 **Framework-friendly** with examples for FastHTML, Flask, Django, and more
- 📦 **Lightweight** with minimal dependencies
- 🔧 **Customizable icon sets** - include only the icons you need

## Installation
```bash
pip install python-lucide
```
This installs the package with a pre-built database of all Lucide icons, ready to use immediately.

## Quick Start
```python
from lucide import lucide_icon

# Get an icon
svg = lucide_icon("house")

# Add CSS classes
svg = lucide_icon("settings", cls="icon icon-settings")

# Customize size
svg = lucide_icon("arrow-up", width="32", height="32")

# Customize colors (stroke for outline, fill for interior)
svg = lucide_icon("heart", stroke="red", fill="pink")

# Customize stroke properties
svg = lucide_icon("chart-line", stroke_width="3", stroke_linecap="round")
```

### Icon Customization
All Lucide icons use `stroke` for their outline color and `fill` for their interior color:
```python
# Looking for how to change colors? Use stroke and fill:
lucide_icon("user", stroke="blue")        # Blue outline
lucide_icon("user", fill="currentColor")  # Inherit color from CSS
lucide_icon("user", stroke="#ff6b6b")     # Hex colors work too
```

## Framework Integration Examples

### FastHTML
```python
from fasthtml.common import *
from lucide import lucide_icon

app, rt = fast_app()

@rt('/')
def get():
    return Titled("Hello Icons",
        H1("Welcome"),
        # Wrap icon output in NotStr to prevent HTML escaping
        NotStr(lucide_icon("house", cls="icon")),
        P("This is a simple FastHTML app with Lucide icons.")
    )

serve()
```

### Flask
```python
from flask import Flask
from lucide import lucide_icon

app = Flask(__name__)

@app.route('/icons/<icon_name>')
def serve_icon(icon_name):
    svg = lucide_icon(icon_name, cls="icon", stroke="currentColor")
    return svg, 200, {'Content-Type': 'image/svg+xml'}
```

### Django
```python
# In your views.py
from django.http import HttpResponse
from lucide import lucide_icon

def icon_view(request, icon_name):
    svg = lucide_icon(icon_name, cls="icon-lg", width="32", height="32")
    return HttpResponse(svg, content_type='image/svg+xml')

# In your templates (as a template tag)
from django import template
from django.utils.safestring import mark_safe
from lucide import lucide_icon

register = template.Library()

@register.simple_tag
def icon(name, **kwargs):
    return mark_safe(lucide_icon(name, **kwargs))
```

### FastAPI
```python
from fastapi import FastAPI
from fastapi.responses import Response
from lucide import lucide_icon

app = FastAPI()

@app.get("/icons/{icon_name}")
def get_icon(icon_name: str, size: int = 24, color: str = "currentColor"):
    svg = lucide_icon(icon_name, width=size, height=size, stroke=color)
    return Response(content=svg, media_type="image/svg+xml")
```

## API Reference

### `lucide_icon()`
Retrieves and customizes a Lucide icon.
```python
lucide_icon(
    icon_name: str,
    cls: str = "",
    fallback_text: str | None = None,
    width: str | int | None = None,
    height: str | int | None = None,
    fill: str | None = None,
    stroke: str | None = None,
    stroke_width: str | int | None = None,
    stroke_linecap: str | None = None,
    stroke_linejoin: str | None = None,
) -> str
```
**Parameters:**
- `icon_name`: Name of the Lucide icon to retrieve
- `cls`: CSS classes to add to the SVG element (space-separated)
- `fallback_text`: Text to display if the icon is not found
- `width`: Width of the SVG element
- `height`: Height of the SVG element
- `fill`: Fill color for the icon
- `stroke`: Stroke color for the icon (outline color)
- `stroke_width`: Width of the stroke
- `stroke_linecap`: How the ends of strokes are rendered ("round", "butt", "square")
- `stroke_linejoin`: How corners are rendered ("round", "miter", "bevel")

**Returns:** SVG string

**Example:**
```python
# Full customization example
icon = lucide_icon(
    "activity",
    cls="icon icon-activity animated",
    width=48,
    height=48,
    stroke="rgb(59, 130, 246)",
    stroke_width=2.5,
    stroke_linecap="round",
    stroke_linejoin="round"
)
```

### `get_icon_list()`
Returns a list of all available icon names.
```python
from lucide import get_icon_list

icons = get_icon_list()
print(f"Available icons: {len(icons)}")
print(icons[:5])  # ['activity', 'airplay', 'alarm-check', ...]
```

## Advanced Usage

### Building a Custom Icon Set
If you want to include only specific icons or use a different version of Lucide:
```bash
# Build with specific icons only
lucide-db -i home,settings,user,heart,star -o custom-icons.db

# Use a specific Lucide version
lucide-db -t 0.350.0 -o lucide-v0.350.0.db

# Build from a file listing icon names
echo -e "home\nsettings\nuser" > my-icons.txt
lucide-db -f my-icons.txt -o my-icons.db
```

### Using a Custom Database
Set the `LUCIDE_DB_PATH` environment variable:
```bash
export LUCIDE_DB_PATH=/path/to/custom-icons.db
python your-app.py
```
Or configure it in your Python code:
```python
import os
os.environ['LUCIDE_DB_PATH'] = '/path/to/custom-icons.db'

from lucide import lucide_icon
# Will now use your custom database
```

## Development
This project uses `uv` for fast dependency management and `pre-commit` for code quality.

### Setup
```bash
# Clone the repository
git clone https://github.com/mmacpherson/python-lucide.git
cd python-lucide

# Create a virtual environment and install dependencies
make env
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pre-commit hooks
make install-hooks

# Run tests
make test
```

### Rebuilding the Icon Database
```bash
# Rebuild with latest Lucide icons
make lucide-db

# Rebuild with specific version
make lucide-db TAG=0.350.0

# Check if version updates are available
make check-lucide-version
```

### Version Checking and Automation
The project includes automated version checking and update capabilities:

```bash
# Check for Lucide version updates and artifact status
make check-lucide-version

# Alternative: Use the CLI command directly
uv run check-lucide-version
```

**Weekly Automation**: The repository automatically checks for new Lucide releases every Monday and creates update PRs when new versions are available.

### Release Process
This project follows a manual release process:

1. **Update version** in `pyproject.toml`:
   ```bash
   # Create release branch
   git checkout -b release/v0.2.0

   # Edit pyproject.toml to bump version
   # version = "0.2.0"

   # Commit and push
   git add pyproject.toml
   git commit -m "Bump version to 0.2.0"
   git push -u origin release/v0.2.0
   ```

2. **Create and merge PR** for the version bump

3. **Trigger publishing workflow**:
   - Go to [Actions](../../actions/workflows/publish.yml)
   - Click "Run workflow"
   - Select the main branch
   - Click "Run workflow"

4. **Automatic publishing**: The `publish.yml` workflow builds and publishes the package to PyPI using trusted publishing.

## How It Works
The package comes with a pre-built SQLite database containing all Lucide icons. When you call `lucide_icon()`, it fetches the icon's SVG from the database and applies your customizations. This approach means:
- **Fast**: Icons are loaded from an efficient SQLite database
- **Offline**: No internet connection required at runtime
- **Customizable**: Build your own database with just the icons you need
- **Maintainable**: Update to newer Lucide versions by rebuilding the database

## License
This project is licensed under the MIT License - see the LICENSE file for details.
The Lucide icons themselves are also MIT licensed - see [Lucide's license](https://github.com/lucide-icons/lucide/blob/main/LICENSE).
