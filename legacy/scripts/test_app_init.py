import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

print("Importing app...")
try:
    import scripts.app as app
    print("App imported successfully.")
    if app.engine is not None:
        print("Engine initialized successfully.")
    else:
        print("Engine initialization FAILED.")
        sys.exit(1)
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Runtime error during import: {e}")
    sys.exit(1)
