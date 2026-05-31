# am170b-final-project

To run the HAVOK method on the H-R model, run `hr_havok.py`.

The tutorial from the PyDMD library running HAVOK on Lorenz
can be found in `lorenz-havok-tutorial.py`.

To run:

```
python3 -m venv venv
source venv/bin/activate
pip install . # Install dependencies
python3 hr_havok.py
```

Contributions:
- Ayush:
  - Researched Hindmarsh-Rose model
  - Implemented HAVOK on H-R
  - Burst prediction and reconstruction quality results (in `hr_havok.py`)
