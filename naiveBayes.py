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
csvRows = loadDataset(r'C:\Users\sikor\archive\flowers.csv')
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
       [(mean(column), stdev(column), len(column))]
    )
 return summaries

meanStdLenByClass = dict()
for (classKey, data) in dataset.items():
   meanStdLenByClass[classKey] = getMeanStdLenForColumns(data)

def calculate_probability(x, mean, stdev):
 exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
 return (1 / (sqrt(2 * pi) * stdev)) * exponent

input()