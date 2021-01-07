import pandas as Pandas
import numpy as Numpy
import matplotlib.pyplot as Pyplot
import time

start_time = time.time()

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
TrainingPath, TestingPath = 'IrisTrainingData.csv', 'IrisTestingData.csv'
SPECIES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

class KNN():
    def __init__(self, k):
        self.k = k

    def TrainingData(self, TrainingPath, ColoumnName='Species'):
        '''
        Load training data
        '''
        
        TrainingCSV = Pandas.read_csv( TrainingPath, names=CSV_COLUMN_NAMES).sample(frac=1).reset_index(drop=True)

        TrainingFS, self.TrainingLS = TrainingCSV, TrainingCSV.pop(ColoumnName)

        self.norm_TrainingFS = (TrainingFS - TrainingFS.min()) / \
                               (TrainingFS.max() - TrainingFS.min())

        return self.norm_TrainingFS, self.TrainingLS

    def TestingData(self, TestingPath, ColoumnName='Species'):
        '''
        Load testing data
        '''
            
        TestingCSV = Pandas.read_csv( TestingPath, names=CSV_COLUMN_NAMES).sample(frac=1).reset_index(drop=True)
            
        TestingFS, self.TestingLS = TestingCSV, TestingCSV.pop(ColoumnName)
            
        self.norm_TestingFS = (TestingFS - TestingFS.min()) / \
            (TestingFS.max() - TestingFS.min())

        return self.norm_TestingFS, self.TestingLS

    def Prediction(self, TestPoint):
        '''
        Prediction the label of each testing
        '''
        Distance = []

        for f in self.norm_TrainingFS.values:
            Distance.append(sum(map(abs, f - TestPoint)))
        
        _ = Pandas.DataFrame({"F": Distance, "L": self.TrainingLS})
        _ = _.sort_values(by='F')['L'][0:self.k].values

        return _

# Initialization
TrainingAccuracy = []
TestingAccuracy = []

for k in range(35):
    knn = KNN(k=k + 1)

    TrainingFS, TrainingLS = knn.TrainingData(TrainingPath)
    TestingFS, TestingLS = knn.TestingData(TestingPath)

#Training Process
    correct = 0  
    for i, TestPoint in enumerate(TrainingFS.values, 0):
        _ = knn.Prediction(TestPoint)
        count = [list(_).count('Iris-setosa'),list(_).count('Iris-versicolor'), list(_).count('Iris-virginica')]
        print('Distribution: {}'.format(count))
        mode = SPECIES[count.index(max(count))]
        if mode == TrainingLS[i]:
            correct += 1
        print('Prediction: {}'.format(mode), 'TEST_LABEL: {}'.format(TrainingLS[i]),)
   
    TrainingAccuracy.append(correct / len(TrainingFS))

#Testing Process
    correct = 0  
    for i, TestPoint in enumerate(TestingFS.values, 0):
        _ = knn.Prediction(TestPoint)
        count = [list(_).count('Iris-setosa'),list(_).count('Iris-versicolor'), list(_).count('Iris-virginica')]
        print('Distribution: {}'.format(count))
        mode = SPECIES[count.index(max(count))]
        if mode == TestingLS[i]:
            correct += 1
        print('Prediction: {}'.format(mode), 'TEST_LABEL: {}'.format(TestingLS[i]),)
    
    TestingAccuracy.append(correct / len(TestingFS))

#Grapich of Training Accuracy with k = 1 to 35
for (i, EachResult) in enumerate(TrainingAccuracy, 0):
    print('k: {}'.format(i + 1), 'Accuracy: {}'.format(EachResult))

Pyplot.figure()
Pyplot.plot(Numpy.arange(0, 35, 1), TrainingAccuracy, color='orange')
Pyplot.plot(Numpy.arange(0, 35, 1), TestingAccuracy, color='g')
Pyplot.legend(('Training Accuracy', 'Testing Accuracy'), loc=3)
Pyplot.title('k - Accuracy')
Pyplot.xlabel('Number of k')
Pyplot.ylabel('Accuracy')
Pyplot.show()

#Grapich of Testing Accuracy with k = 1 to 35
for (i, EachResult) in enumerate(TestingAccuracy, 0):
    print('k: {}'.format(i + 1), 'Accuracy: {}'.format(EachResult))
Pyplot.figure()
Pyplot.plot(Numpy.arange(0, 35, 1), TestingAccuracy, color='g')
Pyplot.title('k - Accuracy')
Pyplot.xlabel('Number of k')
Pyplot.ylabel('Accuracy')
Pyplot.show()

print("--- %s seconds ---" % (time.time() - start_time))
