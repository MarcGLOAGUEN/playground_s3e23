import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv("data/train.csv", index_col='id')

X = train.drop(columns="defects")
y = train.defects

class_weight = y.value_counts(True).to_dict()
skf = StratifiedKFold(n_splits=4, shuffle=True)

params = {
    "n_estimators": [256 ,512, 1024],
    "criterion" : ['gini', 'entropy','log_loss'],
    "max_depth": [2, 4, 8, 16, 32],
    "min_samples_split": [124, 256],
    'min_samples_leaf':[1, 2, 4],
    'class_weight':['balanced','balanced_subsample',class_weight]

}

model = RandomForestClassifier()

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
    {20*"-"}
    
"""


with open('grid_search.txt', 'a') as fichier:
    ma_chaine = text
    fichier.write(ma_chaine)