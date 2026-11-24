import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("_bootstrap.py")), run_name="termux_vision_bootstrap")

from termux_vision.app import main

raise SystemExit(main())
