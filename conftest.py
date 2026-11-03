import os
import sys
import tempfile

# Tests must never write into the real data/traffic_system.db.
# Point the app at a throw-away SQLite file BEFORE any backend module is imported.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="trafficai_test_")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TEST_DB_DIR, "test.db").replace("\\", "/")

# Ensure project root is in sys.path for test discovery and imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
