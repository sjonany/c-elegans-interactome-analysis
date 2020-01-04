# Notebooks should in this directory should import this file on the very first line
# to maek sure that the repo root's utility directories are accessible.
# From https://stackoverflow.com/a/51028921
import sys
import os

module_path = os.path.abspath(os.pardir)
if module_path not in sys.path:
    sys.path.append(module_path)