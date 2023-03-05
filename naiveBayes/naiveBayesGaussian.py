import csv
from math import sqrt
from math import pi
from math import exp
import statistics
import itertools
from sklearn.model_selection import train_test_split

# data prep
def loadDataset(filePath):
    X = list()
    Y = list()
    with open(filePath) as csvFile:
        datasetReader = csv.reader(csvFile, delimiter=',')
        for row in datasetReader:
            if not row:
                continue
            X.append(list(map(float, row[:-1])))
            Y.append(row[-1])
    return (X, Y)

def splitDatasetByClassKey(x_train, y_train):
    dataset = dict()
    for xy in itertools.zip_longest(x_train, y_train):
        if xy[-1] not in dataset.keys():
            dataset[xy[-1]] = list()
        dataset[xy[-1]].append(xy[0])
    return dataset

# data loaded
X, Y = loadDataset(r'C:\Users\sikor\archive\flowers.csv')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

trainDataset = splitDatasetByClassKey(x_train, y_train)

def getMeanStdLenForColumns(dataset):
 summaries = list() 
 for column in zip(*dataset):
    if any(isinstance(t, str) for t in column):
       continue
    summaries.append(
       [statistics.mean(column), statistics.stdev(column), len(column)]
    )
 return summaries

meanStdLenByClass = dict()
for (classKey, data) in trainDataset.items():
   meanStdLenByClass[classKey] = getMeanStdLenForColumns(data)

# used to calculate P(feature) from Gaussian distribution
# based on training set mean and stdev extracted from given class -> P(feature|class) 
def calculate_probability(x, mean, stdev):
 exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
 return (1 / (sqrt(2 * pi) * stdev)) * exponent

# return P(class | X1, X2 ... Xn)
def calculate_class_probabilities(summaries, row):
    probabilities = dict()
    totalRows = sum([summaries[label][0][2] for label in summaries])
    for classKey, items in summaries.items():
        # P(class) for each class
        probabilities[classKey] = summaries[classKey][0][2] / totalRows
        for i in range(len(items)):
            mean, stdev, count = items[i]
            # P(class|X1,X2) = P(X1|class) * P(X2|class) * P(class)
            probabilities[classKey] *= calculate_probability(row[i], mean, stdev)

    return probabilities

# model accuracy
classMismatch = 0
for xy in itertools.zip_longest(x_test, y_test):
    probabilities = calculate_class_probabilities(meanStdLenByClass, xy[:-1][0])  
    keyWithMax = max(probabilities, key=probabilities.get)
    if keyWithMax != xy[-1]:
       classMismatch+=1

print(f'Model accuracy: {len(x_test) - classMismatch}/{len(x_test)}')