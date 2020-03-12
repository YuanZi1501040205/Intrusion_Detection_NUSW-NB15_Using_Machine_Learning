"""train.py: Starter file to run DT,RF,SVM classifier"""


#Example Usage: python train -c two -r 0.33 -m svm -n std -f 25
#Example Usage: python train -c multi -r 0.33 -m svm -n normtype -f 30

__author__      = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"



def main():
    """ The main funtion that parses input arguments, calls the approrpiate
     kmeans method and writes the output image"""

    # !!configure your datasets' path here!!--------------------------------------------------------------------------------
    path1 = '/data/UNSW-NB15/UNSW-NB15_1.csv'
    path2 = '/data/UNSW-NB15/UNSW-NB15_2.csv'
    path3 = '/data/UNSW-NB15/UNSW-NB15_3.csv'
    path4 = '/data/UNSW-NB15/UNSW-NB15_4.csv'
    # -----------------------------------------------------------------------------------------------------------------------

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
                        help="Specify the number of PCA components to train model: 1-45(integer)", metavar="FEATURES")
    args = parser.parse_args()

    #read parametes
    if args.classes is None:
        print("classifier not specified using two classifier")
        print("use the -h option to see usage information")
        classifier = 'two'
        pos_label = 2
    else:
        classifier = args.classes
        pos_label = 9


    if args.rate is None:
        print("ratio of dataset splitting not specified using 0.33")
        print("use the -h option to see usage information")
        rate = 0.33
    else:
        rate = int(args.rate)

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
        n_features = 45
    else:
        n_features = int(args.features)


    Preprocessing = preprocess()
    # Merge 4 cvs file to one dataset with head
    dataset = Preprocessing.merge4datasets(path1, path2, path3, path4)
    # Convert categorical data into discrete values using label encoder
    dataset, inv_attack_cat_mapping = Preprocessing.mapping(dataset)
    # Normalization
    if normtype == 'std':
        dataset = Preprocessing.normalization(dataset, normtype)
    elif normtype == 'maxmin':
        dataset = Preprocessing.normalization(dataset, normtype)
    elif normtype == 'all':
        dataset = Preprocessing.normalization(dataset, 'maxmin')
        dataset = Preprocessing.normalization(dataset, 'maxmin')
    else:
        pass
    #split datset to train and test dataset
    X_train, X_test, Y_train, Y_test = Preprocessing.split(dataset, rate, classifier)
    # PCA analysis, show variance of each features
    X_train, X_test = Preprocessing.pca_analysis(X_train, X_test, n_features)

### Models
    from sklearn.metrics import recall_score
    import sklearn.metrics as metrics

    # Decision Tree
    if model == 'dt':
        from sklearn import tree
        # Create Decision Tree classifer object
        clf = tree.DecisionTreeClassifier()
        # Train Decision Tree Classifer
        clf = clf.fit(X_train, Y_train)
        # Predict the response for test dataset
        Y_pred = clf.predict(X_test)

    # Random forest
    elif model == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        # Create a Gaussian Claasifier
        clf = RandomForestClassifier(n_estimators=100)
        # Train Random Forest Classifer
        clf = clf.fit(X_train, Y_train)
        # Predict the response for test dataset
        Y_pred = clf.predict(X_test)
    # SVM
    elif model == 'svm':
        from sklearn.svm import SVC
        # train SVM model
        svm_model_rbf = SVC(kernel='rbf', C=1).fit(X_train, Y_train)
        #Predict the response for test dataset
        Y_pred = svm_model_rbf.predict(X_test)

    ## Evaluating Model
    # Model Accuracy, how often is the classifier correct?
    from sklearn.metrics import classification_report
    print(inv_attack_cat_mapping)
    print(classification_report(Y_test, Y_pred, labels=[0,1,2,3,4,5,6,7,8,9]))
