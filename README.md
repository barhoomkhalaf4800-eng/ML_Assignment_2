# ML Assignment 2 — Logistic Regression with Five Optimizers

Binary classification on the **Breast Cancer Wisconsin (Diagnostic)** dataset, available via `sklearn.datasets.load_breast_cancer`. This repository includes an exported CSV (569 samples, 30 numeric features and 1 target). **0 = malignant; 1 = benign**. Educational use only, not for clinical diagnosis.

## Notebooks
1. `01_Gradient_Descent.ipynb` — GD, GD + Momentum
2. `02_Stochastic_Gradient_Descent.ipynb` — SGD, SGD + Momentum
3. `03_Mini_Batch_Gradient_Descent.ipynb` — Mini-Batch, Mini-Batch + Momentum
4. `04_RMSprop.ipynb` — RMSprop
5. `05_Adam.ipynb` — Adam

## Run
```bash
pip install -r requirements.txt
jupyter notebook
```
Open each notebook and select **Run All**. Keep `breast_cancer_dataset.csv` in the same directory.

## Method
- 80/20 stratified train/test split (random_state=42).
- StandardScaler fitted on training features only; intercept added manually.
- Sigmoid + Binary Cross-Entropy loss; gradient and all optimizer updates implemented in NumPy.
- Train loss history, test loss, accuracy, precision, recall, F1 and confusion matrices.
- For fair comparison, note the optimizers have different learning rates and epoch counts; results are *not* a controlled equal-compute benchmark.
- The test set is used for final evaluation, not hyperparameter selection.

## Dataset source
Scikit-learn `load_breast_cancer` (Breast Cancer Wisconsin Diagnostic dataset; original data credited to W. H. Wolberg, W. N. Street and O. L. Mangasarian, University of Wisconsin).
