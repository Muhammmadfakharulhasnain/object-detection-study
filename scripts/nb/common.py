import nbformat as nbf
def md(t): return nbf.v4.new_markdown_cell(t.strip("\n"))
def code(t): return nbf.v4.new_code_cell(t.strip("\n"))
SETUP = '''import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
FIG = os.path.abspath("../figures"); RES = os.path.abspath("../results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
VOC = os.path.abspath("../data/VOCdevkit/VOC2007")'''
def save(cells, name):
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, f"notebooks/{name}.ipynb")
