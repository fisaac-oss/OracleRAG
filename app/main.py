import sys
from pathlib import Path

# Add project root directory to sys.path so 'app' and 'src' modules resolve cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.ui import build_ui

def main() -> None:
    """Entry point script for launching the application."""
    build_ui()

if __name__ == "__main__":
    main()
