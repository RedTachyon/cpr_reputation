FROM rayproject/ray-ml
# Python 3.7
RUN python -m pip install --upgrade pip

RUN pip install numba

WORKDIR home/code
