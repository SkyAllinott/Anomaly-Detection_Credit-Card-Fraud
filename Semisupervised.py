import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Load data:
raw_data = pd.read_csv("G:/My Drive/Python Projects/Fraud/full_data.csv")
raw_data['fraud'].value_counts()

# Assume we have a small subset labelled:
seed = 12
labelled, unlabelled = train_test_split(raw_data, test_size=0.99, random_state=seed)

labelled.to_csv('./Fraud/labelled.csv')
unlabelled.to_csv('./Fraud/unlabelled.csv')


# Treat labelled as our training set, and unlabelled as testing set:
train_labels = labelled['fraud']
train_features = labelled.drop(['fraud'], axis=1)
test_labels = unlabelled['fraud']
test_features = unlabelled.drop(['fraud'], axis=1)

# Set hyperparameter grid:
params = {'max_depth': [1, 3, 5, 7],
          'n_estimators': [500, 700, 900],
          'learning_rate': [0.01, 0.03],
          'subsample': [0.6, 0.7],
          'colsample_bytree': [0.6, 0.7]}

# Calculate pos_weight, which weights the anomalies, due to severe class imbalance
train_labels.value_counts()
scale_pos = 9167/833


iso = xgb.XGBClassifier(random_state=seed, scale_pos_weight=scale_pos)

# Grid searches with 5-fold cross validation over the parameter space.
# I use the f1 score as it balances recall and precision. Recall would lead to more FP's (we detect fraud, but it's real)
# which would annoy customers. Precision would lead to more FN's, which would mean we miss fraud, and costs us money.
clf = GridSearchCV(estimator=iso,
                   param_grid=params,
                   scoring='f1',
                   n_jobs=-1,
                   verbose=1)

clf.fit(train_features.to_numpy(), train_labels.to_numpy())
print("Best parameters: ", clf.best_params_)
print("Best f1 score: ", clf.best_score_)

best_parameters = clf.best_params_
iso_best = xgb.XGBClassifier(random_state=seed, **best_parameters)
iso_best.fit(train_features.to_numpy(), train_labels.to_numpy())

predictions = iso_best.predict(test_features.to_numpy())


ConfusionMatrixDisplay.from_predictions(y_pred=predictions, y_true=test_labels, values_format='')
f1_score(test_labels, predictions)
# F1 score of 0.995 is excellent.
