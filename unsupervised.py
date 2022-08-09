import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score

# Loading in "test" set, created in semisupervised.py
raw_data = pd.read_csv("G:/My Drive/Python Projects/Fraud/unlabelled.csv")

# Assuming a known contamination level:
model = IsolationForest(random_state=9, contamination=0.05)

model.fit(raw_data.drop(['fraud'], axis=1))
anomaly = model.predict(raw_data.drop(['fraud'], axis=1))

predictions = pd.DataFrame({'actual': raw_data['fraud'], 'fitted': anomaly})

predictions['fitted'].replace({1:0, -1: 1}, inplace=True)

ConfusionMatrixDisplay.from_predictions(y_pred=predictions['fitted'], y_true=predictions['actual'], values_format='')

# f1 score of only 0.05!
f1_score(y_pred=predictions['fitted'], y_true=predictions['actual'])

