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
- Hafsah:
  - Implemented Lyapunov exponent analysis to demonstrate chaotic behavior in the Hindmarsh–Rose model.
  - Built 40+ parameter sweep experiments to evaluate HAVOK across different dynamical regimes.
