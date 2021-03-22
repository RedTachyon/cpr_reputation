# Common-pool Resource Problems with Reputation

`cpr_reputation/board.py` is a multi-agent game environment. 

## `pytest`, `flake8`, and `black` are dev dependencies that are not installed from requirements.txt or environment.yaml

## CI sets flake8's line-length to 89 currently. If you don't like it go to `.github/workflows/lint-test.yml` and find the flake8 command and change it. 

### Conda
```
conda env create -f environment.yaml
conda activate cpr-reputation
```

### Pip
```
pip install -r requirements.txt
```

### Docker
```
docker build -t cpr-reputation .
docker run -it cpr-reputation bin/bash
```
