import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X = list()
Y = list()

pd.read_csv()

with open(r'C:\Users\sikor\archive\flowers.csv') as csvFile:
    datasetReader = csv.reader(csvFile, delimiter=',')
    for row in datasetReader:
        if not row:
            continue
        X.append(list(map(float, row[:-1])))
        Y.append(row[-1])


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
    % (len(X_test), (y_test != y_pred).sum()))