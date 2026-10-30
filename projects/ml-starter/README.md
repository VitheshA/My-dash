# ML Starter 01: Iris flower classification

**Status:** educational baseline completed. Not original ML research or external validation. AI-assisted draft; review and reproduce the results yourself before emailing it to researchers as your own work.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python -m pytest -q
```

The included Iris dataset has 150 measured flowers (sepal length/width, petal length/width) and three species labels. The program:

1. Makes a **75/25 stratified split**: 112 training examples and 38 held-out examples.
2. Fits a **DummyClassifier** that always predicts the most common training species (the simple baseline).
3. Fits a **LogisticRegression** model with StandardScaler in a single Pipeline. The scaler is fitted only on training data, preventing test leakage.
4. Evaluates once on the holdout test set and reports accuracy and a confusion matrix.
5. Runs 5-fold stratified cross-validation on the training data for a second estimate of stability.

**Key terms:** `X` = measured features, `y` = correct species, `fit` = learn model coefficients from labeled training data, `predict` = use learned coefficients on unseen examples, `accuracy` = fraction of predictions correct. A confusion matrix shows which species the model confused.

The Iris dataset is easy, small, and old. High accuracy demonstrates the code works on this toy dataset, **not** that the model is scientifically novel, generalizes to a new population, or is ready for real research. After you can explain each line, transfer these practices to a serious domain dataset with a clearly defined hypothesis.
