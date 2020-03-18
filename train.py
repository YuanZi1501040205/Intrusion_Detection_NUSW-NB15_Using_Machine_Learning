"""train.py: Starter file to run DT,RF,SVM classifier"""


#Example Usage: python train.py -c two -r 0.33 -m svm -n std -f 41


__author__      = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"



def main():
    """ The main funtion that parses input arguments, calls the approrpiate
     classification algorithms to train"""
    # !!configure your datasets' path here!!--------------------------------------------------------------------------------
    path1 = '/data/UNSW-NB15/UNSW-NB15_1.csv'
    path2 = '/data/UNSW-NB15/UNSW-NB15_2.csv'
    path3 = '/data/UNSW-NB15/UNSW-NB15_3.csv'
    path4 = '/data/UNSW-NB15/UNSW-NB15_4.csv'
    #-----------------------------------------------------------------------------------------------------------------------
    # path1 = '/data/UNSW-NB15/UNSW-NB15_1 (copy).csv'
    # path2 = '/data/UNSW-NB15/UNSW-NB15_2 (copy).csv'
    # path3 = '/data/UNSW-NB15/UNSW-NB15_3 (copy).csv'
    # path4 = '/data/UNSW-NB15/UNSW-NB15_4 (copy).csv'

#     dataset = preprocess.normalization(dataset, normtype)
#
    #Parse input arguments
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("-c", "--classes", dest="classes",
                        help="Specify the classes to classify two, muti", metavar="CLASSES")
    parser.add_argument("-r", "--rate", dest="rate",
                        help="Specify the ratio of dataset splitting: 0.33(decimal)", metavar="RATIO")
    parser.add_argument("-m", "--model", dest="model",
                        help="Specify the model DT, RF, SVM", metavar="MODEL")
    parser.add_argument("-n", "--normtype", dest="normtype",
                        help="Specify the normtype std, maxmin, all(std+maxmin), if not specified default do not normalize", metavar="NORMTYPE")
    parser.add_argument("-f", "--features", dest="features",
                        help="Specify the number of PCA components to train model: 1-41(integer)", metavar="FEATURES")
    args = parser.parse_args()

    #read parametes
    if args.classes is None:
        print("classifier not specified using two classifier")
        print("use the -h option to see usage information")
        classifier = 'two'
    else:
        classifier = args.classes


    if args.rate is None:
        print("ratio of dataset splitting not specified using 0.33")
        print("use the -h option to see usage information")
        rate = 0.33
    else:
        rate = float(args.rate)

    if args.model is None:
        print("Model not specified using default (decision tree)")
        print("use the -h option to see usage information")
        model = 'dt'
    elif args.model not in ['dt', 'rf', 'svm']:
        print("Unknown model, using default (dt)")
        print("use the -h option to see usage information")
        model = 'dt'
    else:
        model = args.model

    if args.normtype is None:
        print("normtype not specified didn't using normalization")
        print("use the -h option to see usage information")
        classifier = 'none'
    else:
        normtype = args.normtype

    if args.features is None:
        print("the number of PCA components not specified using all features")
        print("use the -h option to see usage information")
        n_features = 41
    else:
        n_features = int(args.features)

    import preprocess

    # Merge 4 cvs file to one dataset with head
    after_merge_dataset = preprocess.merge4datasets(path1, path2, path3, path4)

    # Convert categorical data into discrete values using label encoder
    after_mapping_dataset, inv_attack_cat_mapping = preprocess.mapping(after_merge_dataset)

    # Normalization
    if normtype == 'std':
        after_norm_dataset = preprocess.normalization(after_mapping_dataset, normtype, n_features)
    elif normtype == 'maxmin':
        after_norm_dataset = preprocess.normalization(after_mapping_dataset, normtype, n_features)
    elif normtype == 'all':
        dataset = preprocess.normalization(after_mapping_dataset, 'std', n_features)
        after_norm_dataset = preprocess.normalization(after_mapping_dataset, 'maxmin', n_features)
    else:
        pass

    #split datset to train and test dataset
    X_train, X_test, Y_train, Y_test = preprocess.split(after_norm_dataset, rate, classifier)
    # PCA analysis, show variance of each features and choose n components as featuers to train
    X_train, X_test = preprocess.pca_analysis(X_train, X_test, n_features, after_norm_dataset)

### Models
    import time
    start_time = time.time()
    # Decision Tree
    if model == 'dt':
        from sklearn import tree
        # Create Decision Tree classifer object
        clf = tree.DecisionTreeClassifier()
        # Train Decision Tree Classifer
        print('start training decision tree!')
        clf = clf.fit(X_train, Y_train)
        print('finish training decision tree!')

    # Random forest
    elif model == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        # Create a Gaussian Claasifier
        clf = RandomForestClassifier(n_estimators=100)
        # Train Random Forest Classifer
        print('start training random forest!')
        clf = clf.fit(X_train, Y_train)
        print('finish training random forest!')

    # SVM
    elif model == 'svm':
        from sklearn.svm import SVC
        # train SVM model
        print('start training SVM!')
        clf = SVC(kernel='rbf', C=1).fit(X_train, Y_train)
        print('finish training SVM!')

    #running time
    end_time = time.time()
    print(end_time-start_time)

    #Predict the response for test dataset
    Y_pred = clf.predict(X_test)

    ## Evaluating Model
    # Model Accuracy, how often is the classifier correct?
    from sklearn.metrics import classification_report
    if classifier == 'two':
        report = classification_report(Y_test, Y_pred, labels=[0, 1])
        print(report)
    elif classifier == 'multi':
        print(inv_attack_cat_mapping)
        report = classification_report(Y_test, Y_pred, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
        print(report)

# Output experiment results as csv file
    f= open(model+'_'+str(n_features)+'features_'+normtype+'_'+'classification_report.txt',"w+")
    f.write(report)
    f.close()





# #%% debug here
#     path1 = '/data/UNSW-NB15/UNSW-NB15_1 (copy).csv'
#     path2 = '/data/UNSW-NB15/UNSW-NB15_2 (copy).csv'
#     path3 = '/data/UNSW-NB15/UNSW-NB15_3 (copy).csv'
#     path4 = '/data/UNSW-NB15/UNSW-NB15_4 (copy).csv'
#     import  preprocess
#     import pandas as pd
#     import numpy as np
#     from sklearn import preprocessing
#
#     classifier = 'two'
#     rate = 0.33
#     normtype = 'std'
#     model = 'svm'
#     n_features = 20
#     after_merge_dataset = preprocess.merge4datasets(path1, path2, path3, path4)
#     after_mapping_dataset, inv_attack_cat_mapping = preprocess.mapping(after_merge_dataset)
#     after_norm_dataset = preprocess.normalization(after_mapping_dataset, normtype, n_features)
#     X_train, X_test, Y_train, Y_test = preprocess.split(after_norm_dataset, rate, classifier)
#     X_train, X_test = preprocess.pca_analysis(X_train, X_test, n_features, after_norm_dataset)
# %%
# for feature_name in after_mapping_dataset.columns[:-2]:
#     print(feature_name)
#
    # after_mapping_dataset['sport'] = pd.DataFrame(scaled_array)
    #after_norm_dataset = preprocess.normalization(after_mapping_dataset, normtype)

if __name__ == "__main__":
    main()