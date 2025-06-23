#!/usr/bin/env python3
"""Test script for the Lucide update workflow logic."""

import sys
import pathlib

# Add src to path for importing
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from lucide.config import DEFAULT_LUCIDE_TAG
from lucide.dev_utils import (
    get_latest_lucide_version, 
    compare_versions, 
    get_icon_count_from_db,
    check_version_status
)


def main():
    """Test the automation workflow logic."""
    print("🧪 Testing Lucide Update Workflow Logic")
    print("=" * 50)
    
    # Test version checking
    print("📦 Version Check:")
    current_version = DEFAULT_LUCIDE_TAG
    print(f"  Current: {current_version}")
    
    latest_version = get_latest_lucide_version()
    if latest_version:
        print(f"  Latest:  {latest_version}")
        needs_update = compare_versions(current_version, latest_version)
        print(f"  Update needed: {needs_update}")
    else:
        print("  Latest:  ❌ Failed to fetch")
        return 1
    
    # Test database functions
    print("\n📊 Database Check:")
    icon_count = get_icon_count_from_db()
    print(f"  Icon count: {icon_count}")
    
    # Test comprehensive status check
    print("\n🔍 Full Status Check:")
    status = check_version_status()
    print(f"  Database exists: {status.database_exists}")
    print(f"  Database version: {status.database_version}")
    print(f"  Needs update: {status.needs_update}")
    
    print("\n✅ Workflow logic test completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())