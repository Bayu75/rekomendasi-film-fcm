import os
import sys

# Agar "from src.fcm import ..." bisa dipakai saat pytest dijalankan dari root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
