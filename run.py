import os

n_estimators=[50,100,150,200,250]
max_depth=[5,10,15,20]


for n in n_estimators:
    for m in max_depth:
        os.system(f'python basic_ml.py -n{n} -m{m}')