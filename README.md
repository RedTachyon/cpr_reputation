# Common-pool Resource Problems with Reputation

`cpr_reputation/board.py` is a multi-agent game environment. 

## CI sets flake8's line-length to 89 currently. If you don't like it go to `.github/workflows/lint-test.yml` and find the flake8 command and change it. 

### Conda
```
conda env create -f environment.yaml
conda activate cpr-reputation
pytest
```

### Pip
You can also do something like venv or pipenv for isolation.
```
pip install -r requirements.txt
pytest
```

### Docker
```
docker build -t cpr-reputation .
docker run -it cpr-reputation bin/bash
pytest
```
