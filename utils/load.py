import torch.utils.data
import os
import torch
import numpy as np 

class MyData(torch.utils.data.Dataset):
    def __init__(self,path):
        self.path=path
        self.files=os.listdir(path)

    
    def __len__(self):
        return len(self.files)

        
    def __getitem__(self,idx):
        f=open(self.path+"/"+self.files[idx],"r")
        data_=f.readlines()
        length=len(data_)
        data=[]
        label=[]
        for i in range(length):           
            line=data_[i].split(",")[1:-1]

            data.append([float(x) for x in line[:-2]])
                # self.data[-1][3]/=100
                # label=[int(i)*10 for i in line[-3]]
            
                
            if int(line[-1])==0 and int(line[-2])==0:
                # label.append([1,0,0])
                label.append(0)
            elif int(line[-1])==1:
                label.append(1)
            else:
                label.append(2)
            
        return torch.tensor(data),torch.tensor(label)