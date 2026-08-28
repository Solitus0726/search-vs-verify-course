# Package marker for scripts: make scripts a proper package to prevent shadowing by a same-named package in site-packages
# (notebook's `from scripts.xxx import ...` relies on the repo root being first in sys.path)
