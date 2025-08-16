# main.py
"""
University Admission Prediction - Main Entry Point
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from scripts.run_analysis import main

if __name__ == "__main__":
    main()