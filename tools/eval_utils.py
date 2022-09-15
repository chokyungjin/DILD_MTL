

from re import L
import numpy as np

class Evaluator:
    def __init__(self):
        self.clear()

    def clear(self):
        self.Num_GT=[]
        

    def add(self,pred_cls,gt_cls):
        
        pred_cls=np.argmax(pred_cls)
        self.Num_GT.append(pred_cls==gt_cls)

      

    def get(self,clear=True):
        
        acc = np.mean(self.Num_GT)        
        acc*=100

        if clear:
            self.clear()

        return acc



class Evaluator_multi:
    def __init__(self,num_classes):

        self.num_classes = num_classes

        self.clear()
    def clear(self):
    
        self.Num_GT={}
                
        for i in range(self.num_classes):
             self.Num_GT[i]=[]

                
    def add(self,pred_cls,gt_cls):
        
        gt_cls = int(gt_cls)
        pred_cls=np.argmax(pred_cls)        
        self.Num_GT[gt_cls].append(pred_cls==gt_cls)
        

    def get(self,clear=True):
        
        ACCs=[]

        for i in range(self.num_classes):
            print(self.Num_GT[i])
            acc = np.mean(self.Num_GT[i]) 
            acc=acc*100       
            ACCs.append(acc)
          

        if clear:
            self.clear()

        return ACCs



class Single_Evaluator_2:
    def __init__(self,num_class):
                
        self.num_class=num_class
        self.clear()

    def clear(self):

        self.confusion_matrix=np.zeros([self.num_class,self.num_class],) 

    def add(self,preds,gt_cls):
           
        pred = np.argmax(preds)
        self.confusion_matrix[gt_cls,pred]=1 
               

    def get(self,clear=True):
        
        # c = 0 
        TP = self.confusion_matrix[0,0]
        FP = self.confusion_matrix[1,0] + self.confusion_matrix[2,0]
        FN = self.confusion_matrix[0,1] + self.confusion_matrix[0,2]
        TN = self.confusion_matrix[1,1] + self.confusion_matrix[1,2]+self.confusion_matrix[2,1] + self.confusion_matrix[2,2]

        sensi_c0 = TP / (TP + FN)
        speci_c0 = TN / (TN + FP)
        acc_c0 = (TP+TN)/(TP + FN + TN + FP)

        # c = 1

        TP = self.confusion_matrix[1,1]
        FP = self.confusion_matrix[0,1] + self.confusion_matrix[2,1]
        FN = self.confusion_matrix[1,0] + self.confusion_matrix[1,2]
        TN = self.confusion_matrix[0,0] + self.confusion_matrix[0,2]+self.confusion_matrix[2,0] + self.confusion_matrix[2,2]

        sensi_c1 = TP / (TP + FN)
        speci_c1 = TN / (TN + FP)
        acc_c1 = (TP+TN)/(TP + FN + TN + FP)
            
        # c = 1

        TP = self.confusion_matrix[2,2]
        FP = self.confusion_matrix[1,2] + self.confusion_matrix[0,2]
        FN = self.confusion_matrix[2,0] + self.confusion_matrix[2,1]
        TN = self.confusion_matrix[0,0] + self.confusion_matrix[0,1]+self.confusion_matrix[1,0] + self.confusion_matrix[1,1]

        sensi_c2 = TP / (TP + FN)
        speci_c2 = TN / (TN + FP)
        acc_c2 = (TP+TN)/(TP + FN + TN + FP)

        total_acc  = (acc_c0 + acc_c1 + acc_c2)/3 

        if clear:
            self.clear()

        return sensi_c0,sensi_c1,sensi_c2,speci_c0,speci_c1,speci_c2, total_acc*100
        