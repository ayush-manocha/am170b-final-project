# am170b-final-project

Initial code running the H-R model can be found in `hr-havok.py`.

The standalone manual mHAVOK implementation for Lorenz can be found in
`mhavok_lorenz.py`.

The original exploratory notebook for the same Lorenz implementation can be
found in `mHAVOK_lorenz.ipynb`.

The tutorial from the PyDMD library running HAVOK on Lorenz
can be found in `lorenz-havok-tutorial.py`.

To run:

```
python3 -m venv venv
source venv/bin/activate
pip install . # Install dependencies
python3 hr-havok.py
python3 mhavok_lorenz.py

# Faster smoke test for the standalone Lorenz mHAVOK script
python3 mhavok_lorenz.py --quick
```
