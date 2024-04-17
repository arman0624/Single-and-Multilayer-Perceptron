### Environment

It's highly recommended to use a virtual environment (e.g. conda, venv).

Example of virtual environment creation using conda:
```
conda create -n env_name python=3.10
conda activate env_name
python -m pip install -r requirements.txt
```

### Train and predict commands

```
python3 perceptron.py -d newsgroups -f full -m slp
python3 perceptron.py -d newsgroups -f bow -m slp
python3 perceptron.py -d newsgroups -f bi-gram -m slp
python3 perceptron.py -d newsgroups -f tfidf -m slp

python3 perceptron.py -d sst2 -f full -m slp
python3 perceptron.py -d sst2 -f bow -m slp
python3 perceptron.py -d sst2 -f bi-gram -m slp
python3 perceptron.py -d sst2 -f tfidf -m slp

python3 multilayer_perceptron.py -d newsgroups -f bow -m mlp
python3 multilayer_perceptron.py -d sst2 -f bow -m mlp
```

### Commands to run unittests

```
pytest tests/test_perceptron.py
pytest tests/test_multilayer_perceptron.py
```
