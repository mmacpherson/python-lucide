# ruff: noqa: ARG001
import contextlib
import re
import sqlite3
import xml.etree.ElementTree as ET
from unittest import mock

import pytest

from lucide import core, db  # Import db to patch its members
from lucide.config import DEFAULT_ICON_CACHE_SIZE

SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"

# Constants for magic numbers
NUM_CLASSES_EXPECTED_TWO = 2
NUM_CLASSES_EXPECTED_THREE = 3
NUM_CLASSES_EXPECTED_FOUR = 4
NUM_CLASSES_EXPECTED_FIVE = 5
NUM_CLASSES_EXPECTED_SIX = 6
NUM_CLASSES_EXPECTED_EIGHT = 8


# --- Fixture for Mock Database ---
@pytest.fixture
def mock_db_path_fixture(tmp_path):
    """Create a temporary database for testing with various icons."""
    db_file_path = tmp_path / "test-lucide.db"
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE icons (name TEXT PRIMARY KEY, svg TEXT NOT NULL)")

    # Icon 1: Basic with attributes
    xmlns = SVG_NAMESPACE.strip(r"{}")
    svg1 = f'<svg xmlns="{xmlns}" width="24" height="24" viewBox="0 0 24 24">'
    svg1 += '<circle cx="12" cy="12" r="10"></circle></svg>'
    cursor.execute("INSERT INTO icons VALUES (?, ?)", ("circle", svg1))

    # Icon 2: With an existing class attribute and other attributes
    svg2 = (
        f'<svg xmlns="{xmlns}" id="square-icon" '
        f'class="lucide existing-class" '
        'width="50" height="50" '
        'data-foo="bar">'
    )
    svg2 += '<rect x="5" y="5" width="40" height="40"></rect></svg>'
    cursor.execute("INSERT INTO icons VALUES (?, ?)", ("square", svg2))

    # Icon 3: Minimal SVG, no attributes on svg tag initially (except xmlns if added)
    svg3 = f'<svg xmlns="{xmlns}"><path d="M0 0 L10 10"></path></svg>'  # Added xmlns
    cursor.execute("INSERT INTO icons VALUES (?, ?)", ("line", svg3))

    conn.commit()
    conn.close()

    # Patch get_default_db_path within the db module
    with mock.patch.object(db, "get_default_db_path", return_value=db_file_path):
        # The tests will use this path indirectly via get_default_db_path
        yield db_file_path


# Helper function to parse SVG and check attributes/classes
def get_svg_root(svg_string: str) -> ET.Element:
    try:
        return ET.fromstring(svg_string)
    except ET.ParseError as e:
        pytest.fail(f"Failed to parse SVG string: {svg_string}\\nError: {e}")


# --- Test Cases ---


def test_lucide_icon_existing_no_modification(mock_db_path_fixture):
    """Test retrieving an existing icon without modifications."""
    core.lucide_icon.cache_clear()
    icon_str = core.lucide_icon("circle")
    assert isinstance(icon_str, str)

    root = get_svg_root(icon_str)
    assert root.tag == SVG_NAMESPACE + "svg"
    assert root.get("width") == "24"
    assert root.get("height") == "24"
    # Check for child element (specific check for the circle icon)
    assert any(child.tag.endswith("circle") for child in root), "<circle> tag not found"

    # Check for automatic classes
    classes = root.get("class", "").split()
    assert "lucide" in classes
    assert "lucide-circle" in classes
    assert "lucide-circle-icon" in classes


def test_lucide_icon_with_simple_class(mock_db_path_fixture):
    """Test retrieving an icon and adding a simple CSS class."""
    core.lucide_icon.cache_clear()
    icon_str = core.lucide_icon("circle", cls="my-class")
    root = get_svg_root(icon_str)

    classes = root.get("class", "").split()
    assert "my-class" in classes
    assert "lucide" in classes
    assert "lucide-circle" in classes
    assert "lucide-circle-icon" in classes
    assert root.get("width") == "24"  # Original attribute preserved


def test_lucide_icon_with_multiple_classes_in_cls(mock_db_path_fixture):
    """Test adding multiple space-separated classes via cls."""
    core.lucide_icon.cache_clear()
    icon_str = core.lucide_icon("circle", cls="class1 class2 another-class")
    root = get_svg_root(icon_str)
    classes = root.get("class", "").split()
    assert "class1" in classes
    assert "class2" in classes
    assert "another-class" in classes
    assert "lucide" in classes
    assert "lucide-circle" in classes
    assert "lucide-circle-icon" in classes
    assert len(classes) == NUM_CLASSES_EXPECTED_SIX


def test_lucide_icon_with_namespace_handling(mock_db_path_fixture):
    """Test retrieving an icon with focus on namespace and serialization."""
    core.lucide_icon.cache_clear()
    icon_str = core.lucide_icon("circle", cls="my-class")
    root = get_svg_root(icon_str)  # This helper is fine for parsing

    # Check 1: Parsed structure is correct (existing checks)
    classes = root.get("class", "").split()
    assert "my-class" in classes
    assert root.get("width") == "24"  # Original attribute preserved
    assert root.tag == SVG_NAMESPACE + "svg"  # Good check for parsed tag

    # Check 2: Serialized string does not contain unwanted namespace prefixes
    assert "ns0:" not in icon_str, "Serialized SVG should not contain 'ns0:' prefix"
    assert icon_str.strip().startswith("<svg"), (
        "Serialized SVG should start with <svg ...> "
        "after stripping leading/trailing whitespace"
    )
    # Ensure the xmlns attribute is correctly on the svg tag itself
    # if no modifications happen or if it's the root of modification.
    if not ET.fromstring(icon_str).get(
        "class"
    ):  # Example: if only default ns is expected
        assert 'xmlns="http://www.w3.org/2000/svg"' in icon_str

    # A more robust check for the root tag in the string:

    match = re.match(r"<([a-zA-Z0-9_:]+)", icon_str.strip())
    assert match is not None, "Could not find root tag in serialized string"
    assert match.group(1) == "svg", (
        f"Serialized root tag should be 'svg', not '{match.group(1)}'"
    )


def test_lucide_icon_add_existing_class_is_idempotent(
    mock_db_path_fixture,
):
    """Test adding a class that is already present (should be idempotent)."""
    core.lucide_icon.cache_clear()
    icon_str = core.lucide_icon(
        "square", cls="existing-class lucide"
    )  # Both already present
    root = get_svg_root(icon_str)

    classes = root.get("class", "").split()
    class_set = set(classes)
    assert "existing-class" in class_set
    assert "lucide" in class_set
    assert "lucide-square" in class_set
    assert "lucide-square-icon" in class_set
    assert (
        len(class_set) == NUM_CLASSES_EXPECTED_FOUR
    )  # No new classes added, no duplicates


def test_lucide_icon_with_attrs_basic(mock_db_path_fixture):
    """Test retrieving an icon with custom attributes."""
    core.lucide_icon.cache_clear()
    # Test overriding an existing attribute (width) and adding a new one (stroke)
    icon_str = core.lucide_icon("circle", width="32", stroke="red")
    root = get_svg_root(icon_str)

    assert root.get("width") == "32"  # Overridden
    assert root.get("height") == "24"  # Original height preserved
    assert root.get("stroke") == "red"  # New attribute added


def test_lucide_icon_cls_can_add_class_to_svg_without_class(
    mock_db_path_fixture,
):
    """Test adding class via cls param to an SVG without an initial class."""
    core.lucide_icon.cache_clear()
    class_str_to_add = "new-class1 new-class2"
    # "circle" icon is defined in fixture without a class attribute initially
    icon_str = core.lucide_icon("circle", cls=class_str_to_add)
    root = get_svg_root(icon_str)

    classes = root.get("class", "").split()
    assert "new-class1" in classes
    assert "new-class2" in classes
    assert len(set(classes)) == NUM_CLASSES_EXPECTED_FIVE


def test_lucide_icon_cls_appends_to_existing_class(
    mock_db_path_fixture,
):
    """Test cls parameter APPENDS to existing classes on the SVG."""
    core.lucide_icon.cache_clear()
    # "square" icon fixture has "lucide existing-class"
    cls_to_append = "appended-class another-appended"
    icon_str = core.lucide_icon("square", cls=cls_to_append)
    root = get_svg_root(icon_str)

    classes = root.get("class", "").split()
    class_set = set(classes)
    assert "lucide" in class_set  # Original class preserved
    assert "existing-class" in class_set  # Original class preserved
    assert "appended-class" in class_set  # New class appended
    assert "another-appended" in class_set  # New class appended
    assert "lucide-square" in class_set  # Automatic class added
    # Expected: "lucide", "existing-class", "appended-class",
    #           "another-appended", "lucide-square"
    assert len(class_set) == NUM_CLASSES_EXPECTED_SIX
    assert root.get("id") == "square-icon"  # Other original attributes preserved


def test_lucide_icon_automatic_class_generation(mock_db_path_fixture):
    """Test that automatic Lucide classes are added to SVGs based on icon name."""
    core.lucide_icon.cache_clear()

    # Test with different icon names
    icon_names = ["circle", "square"]

    for icon_name in icon_names:
        # Test without cls parameter
        icon_str = core.lucide_icon(icon_name)
        root = get_svg_root(icon_str)
        classes = root.get("class", "").split()
        class_set = set(classes)

        # Verify automatic classes
        assert "lucide" in class_set
        assert f"lucide-{icon_name}" in class_set

        # Test with cls parameter
        icon_str_with_cls = core.lucide_icon(icon_name, cls="custom-class")
        root_with_cls = get_svg_root(icon_str_with_cls)
        classes_with_cls = root_with_cls.get("class", "").split()
        class_set_with_cls = set(classes_with_cls)

        # Verify automatic classes are still there
        assert "lucide" in class_set_with_cls
        assert f"lucide-{icon_name}" in class_set_with_cls
        assert "custom-class" in class_set_with_cls

    # Test with an icon that already has the automatic classes
    # The "square" fixture already has "lucide" class
    icon_str = core.lucide_icon("square")
    root = get_svg_root(icon_str)
    classes = root.get("class", "").split()

    # Count occurrences to make sure classes aren't duplicated
    class_counts = {}
    for cls in classes:
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # Each class should appear exactly once
    assert class_counts.get("lucide", 0) == 1
    assert class_counts.get("lucide-square", 0) == 1


def test_lucide_icon_cls_merging_with_existing_and_param_classes(
    mock_db_path_fixture,
):
    """Test cls parameter appends unique classes to existing SVG classes."""
    core.lucide_icon.cache_clear()
    # "square" icon's original class from fixture: "lucide existing-class"
    # `cls` param will append "cls-provided-class", "new-attr-class",
    #                         "another-new-attr", "dupe-cls"
    # Note: "dupe-cls" is in `cls_param_str` twice, but should only appear once.
    # "existing-class" is also in `cls_param_str` to test idempotency of adding existing
    cls_param_str = (
        "cls-provided-class new-attr-class another-new-attr "
        "dupe-cls dupe-cls existing-class"
    )

    icon_str = core.lucide_icon(
        "square",
        cls=cls_param_str,
        width="100",  # Test an explicit attribute as well
    )
    root = get_svg_root(icon_str)

    classes = root.get("class", "").split()
    class_set = set(classes)

    # Expected classes:
    # From SVG: "lucide", "existing-class"
    # From cls_param_str (unique): "cls-provided-class", "new-attr-class",
    #                              "another-new-attr", "dupe-cls"
    # Total unique: lucide, existing-class, cls-provided-class, new-attr-class,
    #               another-new-attr, dupe-cls
    # 6 classes total.

    assert "lucide" in class_set  # Original
    assert "existing-class" in class_set  # Original (and in cls_param_str)
    assert "cls-provided-class" in class_set  # From cls param
    assert "new-attr-class" in class_set  # From cls param
    assert "another-new-attr" in class_set  # From cls param
    assert "dupe-cls" in class_set  # From cls param (added once)

    assert len(class_set) == NUM_CLASSES_EXPECTED_EIGHT

    assert root.get("width") == "100"  # Explicit attribute
    assert root.get("id") == "square-icon"  # Original attribute preserved


def test_lucide_icon_minimal_svg_add_all(mock_db_path_fixture):
    """Test adding class and attributes to a minimal SVG."""
    core.lucide_icon.cache_clear()
    icon_str = core.lucide_icon(
        "line",  # "line" icon is minimal in fixture
        cls="important-line styled",
        stroke="red",
        stroke_width="3",
        width="16",
        height="16",
    )
    root = get_svg_root(icon_str)

    classes = root.get("class", "").split()
    assert "important-line" in classes
    assert "styled" in classes
    assert "lucide" in classes
    assert "lucide-line" in classes
    assert len(set(classes)) == NUM_CLASSES_EXPECTED_FIVE

    assert root.get("stroke") == "red"
    assert root.get("stroke-width") == "3"
    assert root.get("width") == "16"
    assert root.get("height") == "16"
    assert root.find(SVG_NAMESPACE + "path") is not None  # Check child is preserved


def test_lucide_icon_cls_empty_string_behavior(
    mock_db_path_fixture,
):
    """Test behavior of cls='' parameter."""
    core.lucide_icon.cache_clear()

    # Scenario 1: Icon initially has no class attribute ("circle" icon from fixture)
    # Providing cls="" should result in only automatic classes being added
    icon_str_circle = core.lucide_icon("circle", cls="")
    root_circle = get_svg_root(icon_str_circle)
    classes = root_circle.get("class", "").split()
    assert "lucide" in classes
    assert "lucide-circle" in classes
    assert "lucide-circle-icon" in classes
    assert len(classes) == NUM_CLASSES_EXPECTED_THREE
    assert set(classes) == {"lucide", "lucide-circle", "lucide-circle-icon"}

    # Scenario 2: Icon initially has classes ("square" icon: "lucide existing-class")
    # Providing cls="" should keep existing classes and add automatic ones.
    icon_str_square = core.lucide_icon("square", cls="")
    root_square = get_svg_root(icon_str_square)
    classes = root_square.get("class", "").split()
    assert "lucide" in classes
    assert "existing-class" in classes
    assert "lucide-square" in classes
    assert "lucide-square-icon" in classes
    assert len(classes) == NUM_CLASSES_EXPECTED_FOUR


def test_lucide_icon_not_found_placeholder(mock_db_path_fixture):
    """Test behavior when icon is not found, returns placeholder SVG."""
    core.lucide_icon.cache_clear()
    icon_name = "non-existent-icon-123"
    icon_str = core.lucide_icon(icon_name)

    assert isinstance(icon_str, str)
    root = get_svg_root(icon_str)
    assert root.tag == SVG_NAMESPACE + "svg"
    assert root.get("data-missing-icon") == icon_name
    assert "lucide-placeholder" in root.get("class", "").split()
    text_element = root.find(SVG_NAMESPACE + "text")
    assert text_element is not None
    assert text_element.text.strip() == icon_name


def test_lucide_icon_not_found_with_custom_fallback_text(
    mock_db_path_fixture,
):
    """Test placeholder SVG uses custom fallback text."""
    core.lucide_icon.cache_clear()
    icon_name = "another-missing-icon"
    fallback = "FB Text"
    icon_str = core.lucide_icon(icon_name, fallback_text=fallback)
    root = get_svg_root(icon_str)

    assert root.get("data-missing-icon") == icon_name
    text_element = root.find(SVG_NAMESPACE + "text")
    assert text_element is not None
    assert text_element.text.strip() == fallback


def test_lucide_icon_db_connection_failure_returns_placeholder(
    mock_db_path_fixture,
):
    """Test placeholder is returned if DB connection cannot be established."""
    core.lucide_icon.cache_clear()
    with mock.patch.object(core, "get_db_connection") as mock_get_conn:

        @contextlib.contextmanager
        def no_connection_ctx(*_, **__):  # Use _ and __ to indicate unused args
            yield None  # Simulate connection failure

        mock_get_conn.side_effect = no_connection_ctx

        icon_str = core.lucide_icon("any-icon-name")
        root = get_svg_root(icon_str)
        assert root.get("data-missing-icon") == "any-icon-name"


def test_lucide_icon_db_query_error_returns_placeholder(
    mock_db_path_fixture,
):
    """Test placeholder for SQLite errors during icon query."""
    core.lucide_icon.cache_clear()
    with mock.patch.object(core, "get_db_connection") as mock_get_conn:
        mock_db_conn_obj = mock.MagicMock()
        mock_cursor_obj = mock.MagicMock()
        mock_cursor_obj.execute.side_effect = sqlite3.OperationalError(
            "Simulated query failure"
        )
        mock_db_conn_obj.cursor.return_value = mock_cursor_obj

        @contextlib.contextmanager
        def faulty_query_ctx(*_, **__):  # Use _ and __ to indicate unused args
            yield mock_db_conn_obj

        mock_get_conn.side_effect = faulty_query_ctx

        icon_str = core.lucide_icon("circle")  # Valid icon name, but query will fail
        root = get_svg_root(icon_str)
        assert root.get("data-missing-icon") == "circle"


def test_configurable_cache_size():
    """Test that the LRU cache size is configurable via config."""
    # Test that the function is decorated with a cache of the configured size
    cache_info = core.lucide_icon.cache_info()
    assert cache_info.maxsize == core.DEFAULT_ICON_CACHE_SIZE
    # Verify it's using the config value

    assert cache_info.maxsize == DEFAULT_ICON_CACHE_SIZE


def test_lucide_icon_caching_behavior(mock_db_path_fixture):
    """Test that lucide_icon results are cached based on all relevant args."""
    core.lucide_icon.cache_clear()
    db_access_count = 0

    # Get the original get_db_connection from the db module
    original_get_db_connection = db.get_db_connection

    @contextlib.contextmanager
    def counting_get_db_connection_wrapper(*args, **kwargs):
        nonlocal db_access_count
        # Make sure we're using the mock_db_path_fixture
        new_args = list(args)
        if not new_args:
            new_args.append(mock_db_path_fixture)
        else:
            new_args[0] = mock_db_path_fixture

        # Increment the counter before yielding the connection
        db_access_count += 1

        with original_get_db_connection(*new_args, **kwargs) as conn:
            yield conn

    # Patch at the core module level since that's where it's being called from
    with (
        mock.patch("lucide.core.get_db_connection", counting_get_db_connection_wrapper),
        mock.patch(
            "lucide.core.get_default_db_path", return_value=mock_db_path_fixture
        ),
    ):
        # Call 1: Fetch "circle"
        icon1_str = core.lucide_icon("circle")
        assert db_access_count == 1
        get_svg_root(icon1_str)

        icon2_str = core.lucide_icon("circle")
        assert db_access_count == 1
        assert icon1_str == icon2_str

        icon3_str = core.lucide_icon("circle", cls="class-a")
        assert db_access_count == NUM_CLASSES_EXPECTED_TWO
        root3 = get_svg_root(icon3_str)
        assert "class-a" in root3.get("class", "").split()
        assert icon1_str != icon3_str

        icon4_str = core.lucide_icon("circle", cls="class-b")
        assert db_access_count == NUM_CLASSES_EXPECTED_THREE
        assert icon3_str != icon4_str

        # Test with an explicit attribute (width)
        icon5_str = core.lucide_icon("circle", width="10")
        assert db_access_count == NUM_CLASSES_EXPECTED_FOUR
        root5 = get_svg_root(icon5_str)
        assert root5.get("width") == "10"
        assert icon1_str != icon5_str

        icon6_str = core.lucide_icon("circle", cls="class-a")
        assert db_access_count == NUM_CLASSES_EXPECTED_FOUR
        assert icon3_str == icon6_str

        icon7_str = core.lucide_icon("square")
        assert db_access_count == NUM_CLASSES_EXPECTED_FIVE
        get_svg_root(icon7_str)
        assert icon1_str != icon7_str


def test_get_icon_list_fetches_correctly(mock_db_path_fixture):
    """Test retrieving the list of all available icon names."""
    core.lucide_icon.cache_clear()  # Clear to ensure get_icon_list hits DB
    icon_list = core.get_icon_list()
    assert isinstance(icon_list, list)
    expected_icons = sorted(["circle", "square", "line"])  # From mock_db_path_fixture
    assert icon_list == expected_icons


def test_get_icon_list_db_connection_error(mock_db_path_fixture):
    """Test get_icon_list returns empty list if DB connection fails."""
    core.lucide_icon.cache_clear()

    # Create a dummy connection that returns None to simulate failure
    @contextlib.contextmanager
    def no_connection_ctx(*_, **__):
        yield None

    # Override both the db connection and the default path
    with mock.patch("lucide.core.get_db_connection") as mock_get_conn:
        mock_get_conn.side_effect = no_connection_ctx

        # This ensures the function sees our mocked connection
        icon_list = core.get_icon_list()
        assert icon_list == []


def test_lucide_icon_invalid_svg_from_db_fallback(mock_db_path_fixture):
    """Test fallback if SVG from DB is invalid and cannot be parsed."""
    core.lucide_icon.cache_clear()

    # Override an icon in the DB with invalid SVG content
    conn = sqlite3.connect(mock_db_path_fixture)  # Use the actual path from fixture
    cursor = conn.cursor()
    malformed_svg = "<svg><unclosed-tag"
    cursor.execute("UPDATE icons SET svg = ? WHERE name = ?", (malformed_svg, "circle"))
    conn.commit()
    conn.close()

    # When patching the db path, we should get our malformed SVG back
    with mock.patch(
        "lucide.core.get_default_db_path", return_value=mock_db_path_fixture
    ):
        # Simple case - no attributes or classes
        icon_str = core.lucide_icon("circle")
        # We should get back the exact malformed SVG content that was in the DB
        assert icon_str == malformed_svg, "Should return original SVG on parse error"

        # Now try with a class parameter - should still return original malformed SVG
        icon_str_with_class = core.lucide_icon("circle", cls="test")
        assert icon_str_with_class == malformed_svg, (
            "Should return original SVG on parse error with cls"
        )
