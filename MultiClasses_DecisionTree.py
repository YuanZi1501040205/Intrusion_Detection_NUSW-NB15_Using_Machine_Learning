# %%
from sklearn import tree
import pandas as pd
import numpy as np

#load dataset
train_dataset = pd.read_csv('/home/cougarnet.uh.edu/yzi2/PycharmProjects/Machine_Learning_Intrusion_Detection_NUSW-NB15/dataset/UNSW_NB15_training-set.csv')
test_dataset = pd.read_csv('/home/cougarnet.uh.edu/yzi2/PycharmProjects/Machine_Learning_Intrusion_Detection_NUSW-NB15/dataset/UNSW_NB15_testing-set.csv')

#delete unrelated feature id
train_dataset = train_dataset.drop(['id'], axis=1)
test_dataset = train_dataset.drop(['id'], axis=1)

#mapping nominal features(proto, state, service, attack_cat) to int
proto_mapping = {label:idx for idx, label in enumerate(np.unique(train_dataset['proto']))}
state_mapping = {label:idx for idx, label in enumerate(np.unique(train_dataset['state']))}
service_mapping = {label:idx for idx, label in enumerate(np.unique(train_dataset['service']))}
attack_cat_mapping = {label:idx for idx, label in enumerate(np.unique(train_dataset['attack_cat']))}
train_dataset['proto'] = train_dataset['proto'].map(proto_mapping)
train_dataset['state'] = train_dataset['state'].map(state_mapping)
train_dataset['service'] = train_dataset['service'].map(service_mapping)
train_dataset['attack_cat'] = train_dataset['attack_cat'].map(attack_cat_mapping)

# "debug" inverse mapping to show original feature
# inv_proto_mapping = {v: k for k, v in proto_mapping.items()}
# train_dataset['proto'] = train_dataset['proto'].map(inv_proto_mapping)
# print(proto_mapping)
# %%Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

result = clf.score(X_test, Y_test)