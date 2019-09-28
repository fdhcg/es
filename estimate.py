import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils.load import MyData
from models.nets import RNN

# from models.nets import MLP,RNN

SEQ_LEN=50
BATCH_SIZE=64
DEVICE=torch.device('cuda:0')
rnn=torch.load("gru-bi-h18-l5.pth")
# rnn=RNN(9,18,4,True,60).to(DEVICE)
test_data=MyData("./data/test")
test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,drop_last=False)
h_state = None

def out2file(a,b,f):
    # a,b are 1-D tensors
    if len(a)!=len(b):
        return -1
    
    l1=a.long().cpu().data.numpy().tolist()
    l2=b.long().cpu().data.numpy().tolist()
    l1=" ".join([str(x) for x in l1])
    l2=" ".join([str(x) for x in l2])
    f.write(l1+"\n")
    f.write(l2+"\n\n")
      

rnn.eval()
length=0
error=0
with torch.no_grad():
    error=torch.tensor(0)
    f=open("estimate.txt","w")
    for idx, (x, y) in enumerate(test_loader):
        batch_x = Variable(x).to(DEVICE)
        # batch_y_ = Variable(y).permute(1,0,2)
        batch_y = Variable(y).to(DEVICE) 
        for i in batch_y:
            length+=len(i)
        output=rnn(batch_x,h_state)
        # output,h_state=rnn(batch_x,h_state)
        # h_state = h_state.data
        # print(output)
        _,output=output.max(dim=2)

        output=output.permute(1,0)


        for i in range(len(output)):

            # l1=str(output[i].long().cpu().data.numpy())+"\n"
            # l2=str(batch_y[i].long().cpu().data.numpy())+"\n"
            # f.write(l1)
            # f.write(l2)
            out2file(output[i],batch_y[i],f)


        diff=(output!=batch_y)
        sum_error=sum(sum(diff.cpu().data.numpy()))
        
        error+=sum_error

    error=error.cpu().data.numpy()/(length)
    print('train acc:'+str(100-error*100)+'%')
    f.close()



f=open("estimate.txt","r")
idx=0
while 1:
    l_pre=f.readline()
    
    
    idx+=1
    if l_pre=='\n':
        continue
    elif not l_pre:
        break
    else:
        l_pre=[int(i) for i in l_pre[:-1].split(' ')]
        l_gt=[int(i) for i in f.readline()[:-1].split(' ')]
    x=range(0,len(l_pre),1)
    plt.figure()
    plt.title(str(int(idx/2))+' result')
    plt.plot(x,l_pre,label='predict')
    plt.plot(x,l_gt,label='ground truth')
    plt.ylim(0,2.5)
    plt.yticks([0,1,2])
    plt.xlabel('timestep')
    plt.ylabel('label')
    plt.legend()
    plt.savefig('./plt/{}.png'.format(int((idx+1)/2)),format='png')
    plt.close()
