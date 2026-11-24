"""Tambahkan akar proyek ke sys.path agar `import src` dan AppTest bekerja dari folder mana pun."""
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
