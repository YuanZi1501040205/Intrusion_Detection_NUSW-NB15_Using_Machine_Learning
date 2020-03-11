# %%
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn import preprocessing

### Preprocessing

# load dataset
train_dataset = pd.read_csv('/data/UNSW-NB15/UNSW_NB15_training-set.csv')
test_dataset = pd.read_csv('/data/UNSW-NB15/UNSW_NB15_testing-set.csv')

# delete unrelated feature id and attacks' category label
train_dataset = train_dataset.drop(['id', 'attack_cat'], axis=1)
test_dataset = test_dataset.drop(['id', 'attack_cat'], axis=1)

## Mapping nominal features(proto, state, service, attack_cat) to int

# Mapping train data
proto_mapping = {label:idx for idx, label in enumerate(np.unique(train_dataset['proto']))}
state_mapping = {label:idx for idx, label in enumerate(np.unique(train_dataset['state']))}
service_mapping = {label:idx for idx, label in enumerate(np.unique(train_dataset['service']))}
train_dataset['proto'] = train_dataset['proto'].map(proto_mapping)
train_dataset['state'] = train_dataset['state'].map(state_mapping)
train_dataset['service'] = train_dataset['service'].map(service_mapping)

# Mapping test data
test_dataset['proto'] = test_dataset['proto'].map(proto_mapping)
test_dataset['state'] = test_dataset['state'].map(state_mapping)
test_dataset['service'] = test_dataset['service'].map(service_mapping)

## dealling with missing values(delete all cause low proportion)

# convert inf to Nan then delete sample containing NaN
train_dataset.replace([np.inf, -np.inf], np.nan)
test_dataset.replace([np.inf, -np.inf], np.nan)
train_dataset = train_dataset.dropna()
test_dataset = test_dataset.dropna()

# create after normalization dataframe to store dataframe
train_dataset_normalized = train_dataset.copy(deep=True)
test_dataset_normalized = test_dataset.copy(deep=True)

# "debug" inverse mapping to show original feature
# inv_proto_mapping = {v: k for k, v in proto_mapping.items()}
# train_dataset['proto'] = train_dataset['proto'].map(inv_proto_mapping)
# print(proto_mapping)
# %% Normalization each column based on max-min normalization

# normalize train features
for feature_name in train_dataset_normalized.columns:
    # 1.convert the column value of the dataframe as floats
    col_array = train_dataset[feature_name].values.astype(float)
    col_array = col_array.reshape(-1,1)
    # 2. create a min max processing object
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(col_array)
    train_dataset_normalized[feature_name] = pd.DataFrame(scaled_array)

# normalize test features
for feature_name in X_test.columns:
    # 1.convert the column value of the dataframe as floats
    col_array = test_dataset[feature_name].values.astype(float)
    col_array = col_array.reshape(-1,1)
    # 2. create a min max processing object
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(col_array)
    test_dataset_normalized[feature_name] = pd.DataFrame(scaled_array)

# delete sample containing NaN, These NaN may appear because divided 0 or have inf number
  train_dataset_normalized = train_dataset_normalized.dropna()
  test_dataset_normalized = test_dataset_normalized.dropna()

# Split dataset to train and test
Y_train = train_dataset_normalized.label
X_train = train_dataset_normalized.drop(['label'], axis=1)
Y_test = test_dataset_normalized.label
X_test = test_dataset_normalized.drop(['label'], axis=1)
# %% count how man NaN
for feature_name in X_test.columns:
    count = X_test[feature_name].isna().sum()
    print(count)
# %%Decision Tree
# Create a Gaussian Claasifier
clf = RandomForestClassifier(n_estimators=100)

# Train Random Forest Classifer
clf = clf.fit(X_train, Y_train)
# %%
# Predict the response for test dataset
Y_pred = clf.predict(X_test)
# %%Evaluating Model
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
result = clf.score(X_test, Y_test)

# %% Visualizing important features
feature_cols = []
for val in X_train.columns:
    feature_cols.append(val)
feature_imp = pd.Series(clf.feature_importances_,index=feature_cols).sort_values(ascending=False)
print(feature_imp)


# %%select features
# Choose more importance features
X=X_train[['petal length', 'petal width','sepal length']]  # Removed feature "sepal length"
y=Y_train['species']