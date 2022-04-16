# einsum
> Experimental implementation of torch/onnx einsum

## Install

```bash
pip3 install -U einsum
```

Or develop locally:

```bash
https://github.com/sorenlassen/einsum-experiment ~/einsum
cd ~/einsum
python3 setup.py develop
```

## Usage

```py
import einsum

print('TODO')
```

## Tests

Run `einsum`'s test suite:
```bash
pip3 install pytest
pytest
```

Type check with mypy:
```bash
pip3 install mypy
python3 -m mypy src/einsum/lib.py
```

## Release

To publish a new release to pypi:
```
pip3 install python-semantic-release
semantic-release publish
```

## About
`pyproject.toml` was generated with [mkpylib](https://github.com/shawwn/scrap/blob/master/mkpylib).
`setup.py` was generated with [poetry-gen-setup-py](https://github.com/shawwn/scrap/blob/master/poetry-gen-setup-py).

