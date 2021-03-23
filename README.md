# Common-pool Resource Problems with Reputation

`cpr_reputation/board.py` is a multi-agent game environment. 

### Conda
```
conda env create -f environment.yaml
conda activate cpr-reputation
pytest && pytype cpr_reputation
```

### Pip
You can also do something like venv or pipenv for isolation.
```
pip install -r requirements.txt
pytest && pytype cpr_reputation
```

### Docker
```
docker build -t cpr-reputation .
docker run -it cpr-reputation bin/bash
pytest && pytype cpr_reputation
```
