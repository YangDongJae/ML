from random import seed
import random
import math

class DecisionTree:
    def __init__(self):
        self.class_value = None
        self.rows = None
        
    def set_class_value(self,value):
        self.class_value = value
    
    def get_class_value(self):
        return self.class_value
    
    def set_rows(self, rows):
        self.rows = rows
        
    def get_rows(self):
        return self.rows
    
    def calculate_entropy(self,data_list):
        entropy = sum([data_list[i] * (math.log(data_list[i]) * -1)for i in range(len(data_list))])
        return entropy
    
    def split_dataset(self,dataset,feature,value_range):
        """
    

        Parameters
        ----------
        dataset : table
            data set for split
        feature : column index number(int)
            The Feature which core to classification on the rows
        value_range : float
            standard point to classification 

        Returns
        -------
        split dataset.

        """
        left_list = list()
        right_list = list()
        
        for i in range (len(dataset)):
            if dataset[i][feature] 

seed(1)
dt = DecisionTree()
dataset = list()
for i in range(9):
    test_list = list()
    test_list.append(random.randint(1, 2))
    for j in range (1,6):
        test_list.append(random.random())
    dataset.append(test_list)

print(dt.calculate_entropy(dataset[0][1:]))

        