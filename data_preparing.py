# import os
# import numpy as np
# path=r"C:\Users\fdhcg\Desktop\data"
# files=os.listdir(path)
# lines=[]
# Ratio=0.8
# for file in files:
#     if file.endswith('.txt'):
#         f=open(path+"/"+file,'r')
#         data_tmp=f.readlines()
#         data_tmp=data_tmp[1:]
#         length=len(data_tmp)
#         for x in range(length):
#             line=list(filter(None,data_tmp[x].split(' ')))[1:]
#             try:
#                 if not float(line[11][:-1])>-999.: #and float(line[-1][:-1])<-10:
#             #     line[-1]='1\n'
#             # else:
#             #     line[-1]='0\n'
#                     pass
#                 else:
#                     # line=','.join(line)
#                     lines.append([float(x) for x in line])
#             except:
#                 pass
#         f.close()

# length=int(len(lines)/30)
# data=np.empty([0,12])
# for i in range(length):
#     data_tmp=lines[30*i:30*(i+1)]
#     if sum([int(x[-3]) for x in data_tmp])==0:
#         continue
#     else:
#         data=np.append(data,data_tmp,axis=0)

# length=int(len(data)/30)
# data_train=data[:int(length*Ratio)*30]
# data_test=data[int(length*Ratio)*30:]
# # def resample(s):
# #     # s_=s[:-1].split(',')[-3:]
# #     # if int(s_[0]):
# #         return s*10
# f=open('./data/data_train_t.txt','w')
# for i in data_train:
#     tmp=[str(x) for x in i]
#     tmp=','.join(tmp)+'\n'
#     f.write(tmp)
# f.close()

# f=open('./data/data_test_t.txt','w')
# for i in data_test:
#     tmp=[str(x) for x in i]
#     tmp=','.join(tmp)+'\n'
#     f.write(tmp)
# f.close()




# f=open('./data/data_train_resample.txt','w')
# for i in data_train:
#     f.write(resample(i))
# f.close()       
# f=open('./data/data_test_resample.txt','w')
# for i in data_test:
#     f.write(resample(i))
# f.close()          

#
""" 
1

"""
# import os
# mean=[-0.04063998669696801,
#     0.10361997671969404,
#     -0.4579359237292832,
#     -476.96393581287066,
#     1.9791194501413452,
#     -2.0464025275760767,
#     5.5548869796574465,
#     1.3645020785987474,
#     1.1984994179923505]
# std=[5.456716305233628,
#     5.59756294747778,
#     6.582903006299827,
#     108.26432090063716,
#     33.572993283221116,
#     27.824590728784656,
#     5.496541888580198,
#     2.128661788682395,
#     6.253856443291163,]

# def data_combine_filter(path):

#     files=os.listdir(path)

#     # f=open(r"C:\Users\fdhcg\Desktop\data\data-a.txt","w")
#     for file in files:
#         if file.endswith('.txt'):
#             f1=open(path+"/"+file,'r')
#             tmp=f1.readlines()[1:]
#             length=len(tmp)
#             for x in range(length):
#                 line=list(filter(None,tmp[x].split(' ')))
#                 try:
#                     if float(line[-1])<-30:
#                         for i in range(9):
#                             line[i+1]=str(round((float(line[i+1])-mean[i])/std[i],2))
#                         line=','.join(line)
#                         f.write(line)
#                     else:
#                         pass
#                 except:
#                     pass
#         f1.close()
#     f.close()
# if __name__ == "__main__":
#     path=r"C:\Users\fdhcg\Desktop\data"
#     data_combine_filter(path)
""" 
2

"""
# get max-value and min-value for each columnof the dataset
# import numpy as np
# f=open(r"C:\Users\fdhcg\Desktop\icme2.0\\data\data-a-m10.txt","r")
# data=f.readlines()
# length=len(data)
# gap=0
# for i in range(length):
#     try:
#         data[i-gap]=[float(x) for x in data[i].split(",")[1:10]]
#     except:
#         gap+=1
# data=data[:-gap]
# max_list=[]
# min_list=[]
# for j in range(9):
#     a=[x[j] for x in data]
#     a_mean=np.mean(a)
#     a_var=np.std(a)
#     print("{},{}".format(a_mean,a_var))


""" 
3

"""


# import os
# path=r"C:\Users\fdhcg\Desktop\icme2.0\data\data-m10"
# files=os.listdir(path)
# idx=0
# is_test=False
# for file in files:
#     f=open(path+"/"+file,"r")
#     c=f.readlines()
#     if len(c)>=10:
#         if not is_test:
#             f_w=open(r"C:\Users\fdhcg\Desktop\icme2.0\data\data-train"+"/idx"+str(idx)+".txt","w")
#         else:
#             f_w=open(r"C:\Users\fdhcg\Desktop\icme2.0\data\data-test"+"/idx"+str(idx)+".txt","w")
#         for i in c:
#             f_w.write(i)
#         f_w.close()
#         idx+=1
#         if idx>1799:
#             is_test=True
#             idx=0
#         f.close()
#     else:
#         f.close()

"""
4

"""
# import os
# mean=[-0.04063998669696801,
#     0.10361997671969404,
#     -0.4579359237292832,
#     -476.96393581287066,
#     1.9791194501413452,
#     -2.0464025275760767,
#     5.5548869796574465,
#     1.3645020785987474,
#     1.1984994179923505]
# std=[5.456716305233628,
#     5.59756294747778,
#     6.582903006299827,
#     108.26432090063716,
#     33.572993283221116,
#     27.824590728784656,
#     5.496541888580198,
#     2.128661788682395,
#     6.253856443291163,]

# path=r"C:\Users\fdhcg\Desktop\es\data"
# files=os.listdir(r"C:\Users\fdhcg\Desktop\data")
# idx=0
# for file in files:
#     if file.endswith('.txt'):
#         f1=open(r"C:\Users\fdhcg\Desktop\data"+"/"+file,'r')
#         tmp=f1.readlines()[1:]
#         length=len(tmp)
#         f=open(path+"/"+str(idx)+".txt","w")
#         count=0
#         for i in range(length):
            
#             if count>=50:
#                 idx+=1
#                 f=open(path+"/"+str(idx)+".txt","w")
#                 count=0
#             line=list(filter(None,tmp[i].split(' ')))
#             try:
#                 for j in range(9):
#                     line[j+1]=str(round((float(line[j+1])-mean[j])/std[j],2))
#                 line=','.join(line)
#                 f.write(line)
#                 count+=1
#             except:
#                 f.close()
#                 count=0
#                 f=open(path+"/"+str(idx)+".txt","w")
"""
5

"""
mean=[-0.04063998669696801,
    0.10361997671969404,
    -0.4579359237292832,
    -476.96393581287066,
    1.9791194501413452,
    -2.0464025275760767,
    5.5548869796574465,
    1.3645020785987474,
    1.1984994179923505]
std=[5.456716305233628,
    5.59756294747778,
    6.582903006299827,
    108.26432090063716,
    33.572993283221116,
    27.824590728784656,
    5.496541888580198,
    2.128661788682395,
    6.253856443291163,]
import os
import random
path=r"C:\Users\fdhcg\Desktop\es\data\train"
files=os.listdir(r"C:\Users\fdhcg\Desktop\data")
idx=0
for file in files:
    if file.endswith('.txt'):
        f1=open(r"C:\Users\fdhcg\Desktop\data"+"/"+file,'r')
        tmp=f1.readlines()[1:]
        length=len(tmp)
        f=open(path+"/"+str(idx)+".txt","w")
        count=0
        sp=0
        for i in range(length):
            
            if count>=500:
                sp=0
                f.close()          
                p=random.random()
                if p>0.8:
                    path=path.replace('train','test')
                else:
                    path=path.replace('test','train')
                idx+=1
                f=open(path+"/"+str(idx)+".txt","w")
                count=0

            line=list(filter(None,tmp[i].split(' ')))
            a_=line[-2]
            b_=line[-3]
            if int(a_)==0 and int(b_)==0:
                sp+=0.005
            else:
                sp=-1
            try:

                for j in range(9):
                    line[j+1]=str(round((float(line[j+1])-mean[j])/std[j],2))
                    
                line=','.join(line)
                p_=random.random()
                if p_>sp:
                    f.write(line)
                    count+=1
                else:
                    f.close()
                    f=open(path+"/"+str(idx)+".txt","w")
                    count=0

            except:
                f.close()

                count=0
                f=open(path+"/"+str(idx)+".txt","w")
            


    