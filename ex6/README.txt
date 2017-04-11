Ping Kamila (on slack or by email, skamila@ethz.ch) if you need any clarification.

Requirements
------------

We are using scipy to make things easier. Install with:
```
pip3 install --user scipy
```
Or use a [virtualenv](https://docs.python.org/3/tutorial/venv.html):
```
pyvenv ./venv
pip install -r requirements.txt
```

How to use
----------

Run `make` :-)

Note: I have not included train_*_full.txt in git, because those datafiles are ~100M each. They are in `/.gitignore`, but may be synced by floobits.

What is where
-------------

Some TA-provided scripts have been abused here (so if something is not explained, it's theirs). Our stuff:

- `embeddings.py`: Compute GloVe word embeddings. Kamila can explain what is happening there (hopefully).

TODO:
-----

1. GloVe word embeddings. TODO(kamila): finish.
2. Load the training tweets and the built GloVe word embeddings. Using the word embeddings, CONSTRUCT a FEATURE REPRESENTATION of each training tweet (by averaging the word vectors over all words of the tweet).
3. Train a Linear Classifier: Train a linear classifier (e.g. logistic regression or SVM) on your constructed features, using the scikit learn library. Recall that the labels indicate if a tweet used to contain a :) or :( smiley.
4. Prediction: Predict labels for all tweets in the test set.
