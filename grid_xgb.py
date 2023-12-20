import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv("data/train.csv", index_col='id')

X = train.drop(columns="defects")
y = train.defects

skf = StratifiedKFold(n_splits=4, shuffle=True)

params = {
    'learning_rate': [0.1, 0.2, 0.3],
    'n_estimators': [256, 512, 1024],
    'max_depth': [1, 2, 4, 8],
    'min_child_weight': [128,256,512,1024],
    'max_bin' : [128, 256, 512]
}

model = XGBClassifier()

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