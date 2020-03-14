"""preprocess.py: tool file to prepare UNSW_NB15 dataset to train"""


#Example Usage: python preprocess.py


__author__      = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"


# Import APIs
import pandas as pd
import numpy as np



def merge4datasets(path1, path2, path3, path4):

    """ The merge4datasets funtion prepare dataset for
     two classifier take 4 dataset input path add head and merge them to one."""
    import pandas as pd
    # load dataset
    column_names = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smean", "dmean", "trans_depth", "response_body_len", "sjit", "djit", "stime", "ltime", "sinpkt", "dinpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"]
    dataset1 = pd.read_csv(path1)
    dataset2 = pd.read_csv(path2)
    dataset3 = pd.read_csv(path3)
    dataset4 = pd.read_csv(path4)
    dataset1.columns = column_names
    dataset2.columns = column_names
    dataset3.columns = column_names
    dataset4.columns = column_names
    #merge
    dataset = dataset1.append(dataset2, ignore_index=True)
    dataset = dataset.append(dataset3, ignore_index=True)
    dataset = dataset.append(dataset4, ignore_index=True)
    for row in range(len(dataset['label'])):
        if dataset.at[row, 'label'] == 0:
            dataset.at[row, 'attack_cat'] = 'Normal'
    return dataset

# mapping

def mapping(dataset):
    """ delete unrelated feature id and attacks' category label and Mapping nominal features(proto, state, service, attack_cat) to int"""
    import numpy as np

    # delete unrelated feature id and attacks' category label
    after_drop_dataset = dataset.drop(columns=['srcip', 'dstip', 'ct_flw_http_mthd', 'is_ftp_login', 'stime', 'ltime'])

    after_drop_dataset.isna().sum()
    ## Mapping nominal features(proto, state, service, attack_cat) to int

    # Mapping categories data
    proto_mapping = {label: idx for idx, label in enumerate(np.unique(after_drop_dataset['proto']))}
    state_mapping = {label: idx for idx, label in enumerate(np.unique(after_drop_dataset['state']))}
    service_mapping = {label: idx for idx, label in enumerate(np.unique(after_drop_dataset['service']))}
    attack_cat_mapping = {label: idx for idx, label in enumerate(np.unique(after_drop_dataset['attack_cat']))}
    after_drop_dataset['proto'] = after_drop_dataset['proto'].map(proto_mapping)
    after_drop_dataset['state'] = after_drop_dataset['state'].map(state_mapping)
    after_drop_dataset['service'] = after_drop_dataset['service'].map(service_mapping)
    after_drop_dataset['attack_cat'] = after_drop_dataset['attack_cat'].map(attack_cat_mapping)

    ## dealling with missing values(delete all cause low proportion)
    # convert inf to Nan then delete sample containing NaN
    after_drop_dataset.replace([np.inf, -np.inf], np.nan)
    after_drop_dataset = after_drop_dataset.dropna()

    # remapping back the attack categories
    inv_attack_cat_mapping = {v: k for k, v in attack_cat_mapping.items()}
    # "debug" inverse mapping to show original feature
    # inv_proto_mapping = {v: k for k, v in proto_mapping.items()}
    # train_dataset['proto'] = train_dataset['proto'].map(inv_proto_mapping)
    # print(proto_mapping)
    return(after_drop_dataset, inv_attack_cat_mapping)

def normalization(dataset, normtype, n_features):

    """Normalize each column based on max-min normalization or standard scalar"""
    from sklearn import preprocessing
    import pandas as pd

    # 1.convert the whole dataframe value of the dataframe as floats, creating NaNs when necessary
    dataset = dataset.apply(pd.to_numeric, args=('coerce',))

    # convert inf to Nan then delete sample containing NaN
    dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna()

    if normtype == "maxmin":
       # normalize train features with maxmin function
       for feature_name in dataset.columns[:n_features]:
           # 1.convert the column value of the dataframe as floats
           col_array = dataset[feature_name].values.astype(float)
           col_array = col_array.reshape(-1, 1)
           # 2. create a min max processing object
           min_max_scaler = preprocessing.MinMaxScaler()
           scaled_array = min_max_scaler.fit_transform(col_array)
           dataset[feature_name] = pd.DataFrame(scaled_array)
    elif normtype == "std":
       # normalize train features with standard function
       for feature_name in dataset.columns[:n_features]:
           col_array = dataset[feature_name].values.astype(float)
           col_array = col_array.reshape(-1, 1)
           # 2. create a std processing object
           std_scaler = preprocessing.StandardScaler()
           scaled_array = std_scaler.fit_transform(col_array)
           dataset[feature_name] = pd.DataFrame(scaled_array)
    else:
        print("normalize failed, please check your data!")
        pass

    # delete sample containing NaN, These NaN may appear because divided 0 or have inf number
    dataset = dataset.dropna()
    # # %% count how man NaN(debug)
    # for feature_name in X_test.columns:
    #     count = X_test[feature_name].isna().sum()
    #     print(count)
    return(dataset)


def split(dataset, rate, classifier):
    """Choose two or Multi classes classifier and split dataset with assigned rate"""
    from sklearn.model_selection import train_test_split
    if classifier == 'two':
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.drop(['label', 'attack_cat'], axis=1), dataset.label, test_size=rate, random_state=0)
    elif classifier == 'multi':
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.drop(['label', 'attack_cat'], axis=1), dataset.attack_cat, test_size=rate, random_state=0)
    return(X_train, X_test, Y_train, Y_test)



def pca_analysis(X_train, X_test, n_features, after_norm_dataset):
    """ The funtion that analysis importance features with PCA"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components = n_features)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    i = 0
    for col in after_norm_dataset.columns[0:n_features]:
        print(col+' ', explained_variance[i])
        i = i + 1
    return(X_train, X_test)
