# Intrusion_Detection_NUSW-NB15_Using_Machine_Learning

Intrusion_Detection_NUSW-NB15_Using_Machine_Learning is project implemented PCA and three classification algorithms: Decision Tree; Random Forest; SVM with [scikit-learn](https://scikit-learn.org/stable/) on the public Intrusion Detection [USW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/).

### Installation

Install the dependencies.

```sh
$ cd ~
$ git clone https://github.com/YuanZi1501040205/Intrusion_Detection_NUSW-NB15_Using_Machine_Learning.git
$ pip install -r requirements.txt
```

### Experiment Design

| Arguments |Abbreviation |Value |
| ------ |--------|------ |
| Classes:Do binary classification or multiple classes(attack categories) |c |two, multi |
| Rate: the ratio of dataset splitting |r |0.33(0-1decimal) |
| Model:Decision tree, Random Forest, SVM |m |dt, rf, svm|
| Features: the number of features used for training |f |41(0-41integer) |
| Normtype: do standard Normalization or MaxMin Normalization |n |41(0-41integer) |
Example Usage: python train.py -c multi -r 0.33 -m svm -n std -f 25

Edit runme.sh file as you design like following:

```sh
python train.py -c two -r 0.33 -m dt -n std -f 25
python train.py -c multi -r 0.33 -m dt -n std -f 25
python train.py -c two -r 0.33 -m rf -n std -f 25
python train.py -c multi -r 0.33 -m rf -n std -f 25
python train.py -c two -r 0.33 -m svm -n std -f 25
python train.py -c multi -r 0.33 -m svm -n std -f 25
```


# Train!

```sh
$ cd ~
$ ./runme.sh
```
