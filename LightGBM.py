# train with LightGBM
import pandas as pd
import lightgbm as lgb

# load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

X_train = train.drop(['Survived','PassengerId','Name','Ticket','Cabin'], axis=1)
y_train = train['Survived']

X_test = test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

# preprocess data
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

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)

# optimize parameters
params = {'max_depth': 2, 'learning_rate': 1, 'objective': 'binary', 'metric': 'binary_logloss'}
cv_results = lgb.cv(params=params, train_set=lgb_train, nfold=3, num_boost_round=5, early_stopping_rounds=10, metrics='binary_logloss', seed=123)

# train model
model = lgb.train(params=params, train_set=lgb_train, num_boost_round=5)

# predict
y_pred = model.predict(X_test)
y_pred = [1 if x > 0.5 else 0 for x in y_pred]

# save results
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})
submission.to_csv('submissions/LightGBM.csv', index=False)

