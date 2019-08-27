import os

import pandas as pd
from sklearn.metrics import classification_report
from tpot import TPOTClassifier

TRAIN_DATA_PATH = os.path.sep.join(['.', 'data', 'mnist_train.csv'])
TEST_DATA_PATH = os.path.sep.join(['.', 'data', 'mnist_test.csv'])
OUTPUT_PATH = os.path.sep.join(['.', 'output'])

print('[INFO] Loading MNIST dataset.')
train = pd.read_csv(TRAIN_DATA_PATH)
test = pd.read_csv(TEST_DATA_PATH)

y_train = train['label'].astype('int')
y_test = test['label'].astype('int')

X_train = train.drop('label', axis=1).astype('int')
X_test = test.drop('label', axis=1).astype('int')

print('[INFO] Optimization started...')
optimizer = TPOTClassifier(generations=5, population_size=50, cv=5, random_state=77, verbosity=2, n_jobs=1)
optimizer.fit(X_train, y_train)
score = optimizer.score(X_test, y_test)

print(f'[INFO] Score on the test set: {score:.2f}%')

with open(os.path.sep.join([OUTPUT_PATH, 'report.txt']), 'w') as f:
    f.write(classification_report(y_test, optimizer.predict(X_test)))

optimizer.export(os.path.sep.join([OUTPUT_PATH, 'exported_pipeline.py']))
