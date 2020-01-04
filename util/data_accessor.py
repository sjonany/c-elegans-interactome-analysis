"""
utility methods for getting absolute paths to various files in root's data/ folder.
"""

import os

def get_data_file_abs_path(filename):
  return os.path.abspath(os.path.join(os.pardir, 'data', filename))