import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NJORD_ROOT = REPO_ROOT / "njord"
repo_path = str(REPO_ROOT)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

for import_root in (REPO_ROOT, NJORD_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
