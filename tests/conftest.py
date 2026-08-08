"""conftest — shared pytest fixtures."""
import sys
from pathlib import Path

# Ensure project root is in sys.path so all imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))
