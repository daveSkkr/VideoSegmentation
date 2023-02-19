import csv
from math import sqrt
from math import pi
from math import exp

# data prep
def loadDataset(filePath):
    dataset = list()
    with open(filePath) as csvFile:
        datasetReader = csv.reader(csvFile, delimiter=',')
        for row in datasetReader:
            if not row:
                continue
            formated = list(map(float, row[:-1])) + [row[-1]]
            dataset.append(formated)
    return dataset

def splitDatasetByClassKey(datasetRows):
    dataset = dict()
    for row in datasetRows:
        if row[-1] not in dataset.keys():
            dataset[row[-1]] = list()
        dataset[row[-1]].append(row[:-1])
    return dataset

# data loaded
csvRows = loadDataset(r'C:\Users\sikor\żeluś\archive\flowers.csv')
dataset = splitDatasetByClassKey(csvRows)

# get statistics from data columns
def mean(numbers):
 return sum(numbers)/float(len(numbers))

def stdev(numbers):
 avg = mean(numbers)
 variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
 return sqrt(variance)

def getMeanStdLenForColumns(dataset):
 summaries = list() 
 for column in zip(*dataset):
    if any(isinstance(t, str) for t in column):
       continue
    summaries.append(
       [mean(column), stdev(column), len(column)]
    )
 return summaries

meanStdLenByClass = dict()
for (classKey, data) in dataset.items():
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

probabilities = calculate_class_probabilities(meanStdLenByClass, csvRows[0])          

input()