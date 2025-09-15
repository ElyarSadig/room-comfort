Make sure python is installed in your machine and create virtual environment for the project after cloning

```python
py -m venv venv
```

Make sure to activate the virtual environment then install the necessary dependencies:

```python
pip install -r requirements.txt
```

First Run the combiner to gather all data from csv files for each room to a combined csv:

```python
py combiner.py
```

You can then start running and training models with this command:

```python
py knn.py
```

```python
py random_forest.py
```

```python
py linear_regression.py
```

```python
py decision_trees.py
```
