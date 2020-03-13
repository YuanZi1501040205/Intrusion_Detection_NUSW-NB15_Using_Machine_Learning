#!/bin/bash 

python train.py -c two -r 0.33 -m dt -n std -f 41
python train.py -c multi -r 0.33 -m dt -n std -f 41
python train.py -c two -r 0.33 -m rf -n std -f 41
python train.py -c multi -r 0.33 -m rf -n std -f 41
python train.py -c two -r 0.33 -m svm -n std -f 41
python train.py -c multi -r 0.33 -m svm -n std -f 41

python train.py -c two -r 0.33 -m dt -n maxmin -f 41
python train.py -c multi -r 0.33 -m dt -n maxmin -f 41
python train.py -c two -r 0.33 -m rf -n maxmin -f 41
python train.py -c multi -r 0.33 -m rf -n maxmin -f 41
python train.py -c two -r 0.33 -m svm -n maxmin -f 41
python train.py -c multi -r 0.33 -m svm -n maxmin -f 41

python train.py -c two -r 0.33 -m dt -n all -f 41
python train.py -c multi -r 0.33 -m dt -n all -f 41
python train.py -c two -r 0.33 -m rf -n all -f 41
python train.py -c multi -r 0.33 -m rf -n all -f 41
python train.py -c two -r 0.33 -m svm -n all -f 41
python train.py -c multi -r 0.33 -m svm -n all -f 41

python train.py -c two -r 0.33 -m dt -n std -f 25
python train.py -c multi -r 0.33 -m dt -n std -f 25
python train.py -c two -r 0.33 -m rf -n std -f 25
python train.py -c multi -r 0.33 -m rf -n std -f 25
python train.py -c two -r 0.33 -m svm -n std -f 25
python train.py -c multi -r 0.33 -m svm -n std -f 25

