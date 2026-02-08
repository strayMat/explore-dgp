import os
from pathlib import Path

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DIR2ROOT = Path(__file__).parent.parent.absolute()

DIR2DATA = DIR2ROOT / "data"
