"""Build (and optionally execute) the notebooks. usage: python scripts/build_notebooks.py [01 02 ...] [--run]"""
import os, sys, glob
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root); sys.path.insert(0, os.path.join(root, "scripts", "nb"))
args = [a for a in sys.argv[1:] if a != "--run"]
for f in sorted(glob.glob("scripts/nb/nb*.py")):
    if args and os.path.basename(f)[2:4] not in args: continue
    exec(open(f).read(), {"__name__": "nbbuild"})
if "--run" in sys.argv:
    import nbformat
    from nbclient import NotebookClient
    for f in sorted(glob.glob("notebooks/*.ipynb")):
        if args and os.path.basename(f)[:2] not in args: continue
        nb = nbformat.read(f, 4)
        NotebookClient(nb, timeout=1500, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).execute()
        nbformat.write(nb, f); print("executed", f, flush=True)
