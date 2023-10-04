# MLOps Course MIPT

## Setup

To setup only the necessary dependencies, run the following:

```
poetry install --without dev
```

If you want to use `pre-commit`, install all the dependencies:

```
poetry install
```

## Run experiments

To train and evaluate the chosen model, run:

```
poetry run python3 main.py
```

If you only want to train the model and save it afterwards, run:

```
poetry run python3 mlops/train.py
```

If you only want to infer a previously trained model, make sure you've placed the
checkpoint in `saved_models/` and then run

```
poetry run python3 mlops/infer.py
```
