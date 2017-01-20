import unittest
from databaseProxy import DatabaseProxy

class TestDatabaseProxy(unittest.TestCase):
    def test_databaseProxyConstructor(self):
        try:
            dbproxy = DatabaseProxy()
            self.assertTrue(True)
        except:
            self.assertTrue(False)
#basic
    def test_getTestAndTrainingData1(self):
        dbproxy = DatabaseProxy()
        testData, testLabels, trainingData, trainingLabels = dbproxy.getTestAndTrainingData()
        self.assertEqual(len(testData), len(testLabels))
        self.assertEqual(len(trainingData), len(trainingLabels))
        print(len(trainingData), len(testData))
        self.assertTrue(len(trainingData) > len(testData))

#testing sizes
    def test_getTestAndTrainingData2(self):
        dbproxy = DatabaseProxy()
        self.assertTrue(True)
#testing as image
    def test_getTestAndTrainingData3(self):
        dbproxy = DatabaseProxy()
        self.assertTrue(True)
#testing as raw
    def test_getTestAndTrainingData4(self):
        dbproxy = DatabaseProxy()
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
