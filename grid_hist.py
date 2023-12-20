import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv("data/train.csv", index_col='id')

X = train.drop(columns="defects")
y = train.defects

class_weight = y.value_counts(True).to_dict()

skf = StratifiedKFold(n_splits=4, shuffle=True)

params = {
    'learning_rate': [0.01, 0.1, 0.001],
    'max_depth': [None, 2, 4, 8, 16, 32],
    'min_samples_leaf': [4, 8, 16, 32, 64, 128, 256],
    'max_bins': [16, 32, 64, 128, 256, 512],
    'scoring': ['loss', 'roc_auc'],
    'class_weight': ['balanced', class_weight]
}

model = HistGradientBoostingClassifier()

grid = GridSearchCV(
    model,
    param_grid=params,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=20,
    return_train_score=True
)

grid.fit(X, y)

print(grid.best_estimator_)
print(grid.best_score_)

text = f"""
    {grid.best_estimator_}

    Score : {grid.best_score_:.4f}
    {20 * "-"}

"""