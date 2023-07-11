import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

X_train = train.drop(['Survived','PassengerId','Name','Ticket','Cabin'], axis=1)
y_train = train['Survived']

X_test = test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

# in gender column do if male then 1 else 0
mapping = {'male': 1, 'female': 0}
X_train['Sex'] = X_train['Sex'].map(mapping)
X_test['Sex'] = X_test['Sex'].map(mapping)

mapping = {'C': 0, 'Q': 1, 'S': 2}
X_train['Embarked'] = X_train['Embarked'].map(mapping)
X_test['Embarked'] = X_test['Embarked'].map(mapping)

# fill missing values
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)

X_train['Fare'].fillna(X_train['Fare'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

# drop rows with missing values on X and y
X_train.dropna(inplace=True)
y_train = y_train[X_train.index]

X_test.dropna(inplace=True)


# train model

model = LogisticRegression()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# save results
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})
submission.to_csv('submissions/LogReg.csv', index=False)


