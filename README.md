# einsum-experiment
Experimental implementation of torch/onnx einsum

pyproject.toml was filled in following [setuptools quickstart instructions](https://setuptools.pypa.io/en/latest/userguide/quickstart.html),
which makes it possible to generate a distribution with
```bash
python3 -m pip build
python3 -m build
```
for what it's worth.

For now, a self test runs if lib.py is run as the main module:
```bash
python3 src/einsum/lib.py
```
