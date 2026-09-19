"""
main_new.py -- main.py + RT protocol before training (main.py untouched).

  * adds 'rt' to --label_mapping choices
  * loads trainer.<algorithm>.server_new when it exists, else trainer.<algorithm>.server

Usage:
  python main_new.py --algorithm=GeFL_gan_pacfl_iid --label_mapping=rt ...
"""
import importlib
import importlib.util

import main as base

base.IMPLEMENTED_LM.append("rt")


def _import_module(name):
    if name.endswith(".server") and importlib.util.find_spec(name + "_new"):
        name += "_new"
    return importlib.import_module(name)


base.import_module = _import_module

if __name__ == "__main__":
    base.main()
