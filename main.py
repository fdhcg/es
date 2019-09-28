import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils.load import MyData

from models.nets import RNN
import sys
import torch.nn.utils.rnn as rnn_utils
from tensorboardX import SummaryWriter
stdout = sys.stdout
writer=SummaryWriter()
# Hyper Parameter
EPOCH=1000

BATCH_SIZE=64

LR=0.512
# LR=0.001
DEVICE=torch.device('cuda:0')
# DEVICE=torch.device('cpu')
INPUT_SIZE=9
HIDDEN_SIZE=18
BIDIRE=True
SL=60
NUM_LAYER=5
IS_LOAD=False
MOD="gru-bi-h"+str(HIDDEN_SIZE)+"-l"+str(NUM_LAYER)+".pth"



def adjust_learning_rate(optimizer,epoch):
    lr=LR*(0.5**(epoch//200))
    for param_group in optimizer.param_groups:
        param_group['lr']=lr


train_data=MyData("./data/train")
train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,drop_last=False)
test_data=MyData("./data/test")
test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,drop_last=False)


loss_fun=nn.CrossEntropyLoss().to(DEVICE)
if not IS_LOAD:
    rnn=RNN(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYER,BIDIRE,SL).to(DEVICE)
else:
    rnn=torch.load(MOD)
# optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
optimizer = torch.optim.SGD(rnn.parameters(),lr=LR,momentum=0.8,weight_decay=1e-5)

# rnn.train()
h_state = None


for epoch in range(EPOCH):
    adjust_learning_rate(optimizer,epoch)
    rnn.train()
    for i,(x,y) in enumerate(train_loader):


        batch_x = Variable(x).to(DEVICE) # 256*50*9
        batch_y = Variable(y).to(DEVICE) # 256*50

        #y_pack = rnn_utils.pack_padded_sequence(batch_y, l, batch_first=True)#.permute(1,0,2)

        output=rnn(batch_x,h_state).permute(1,2,0) #256*3*50
        # output,h_state=rnn(batch_x,h_state)
        # h_state = h_state.data
        # _, labels = batch_y.max(dim=2)

        loss=loss_fun(output,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%1000:
            stdout.write('\rEpoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i*BATCH_SIZE+len(x), len(train_loader.dataset),
                           100. * (i+1)/ len(train_loader), loss.item()))

    stdout.write('\n')


    if (epoch+1)%10==0:
        # h_state_ = None
        # rnn.eval()
        # with torch.no_grad():
        #     error=torch.tensor([0])
        #     for i, (x, y) in enumerate(train_loader):
        #         batch_x_ = Variable(x).to(DEVICE)
        #         batch_y_ = Variable(y).permute(1,0,2)
        #         output_,h_state_=rnn(batch_x_,h_state_)
        #         h_state_ = h_state_.data
        #         # print(output)
        #         sum_error=sum(sum(sum(torch.abs((batch_dec(output_).long().cpu()-batch_y_)))))
        #         error+=sum_error
        #     error=error.cpu().data.numpy()/(len(train_data)*64*2)
        #     print('train acc:'+str(100-error*100)+'%')
        h_state_ = None

        rnn.eval()
        with torch.no_grad():
            error=torch.tensor(0)
            length=0
            for i, (x,y) in enumerate(test_loader):
               
                batch_x_ = Variable(x).to(DEVICE)
                
                batch_y_ = Variable(y).to(DEVICE) # 64*50
                for i in batch_y_:
                    length+=len(i)
                
                
                #y_pack = rnn_utils.pack_padded_sequence(batch_y, l, batch_first=True)#.permute(1,0,2)

                output_=rnn(batch_x_,h_state_) #50*64*3
                # output_,h_state_=rnn(batch_x_,h_state_)
                # h_state_ = h_state_.data
                # print(output)
                _,output_=output_.max(dim=2)
                diff=(output_.permute(1,0)!=batch_y_)
                # print((batch_dec(output_).long().cpu()))
                # print(batch_y_)
                sum_error=sum(sum(diff.cpu().data.numpy()))
                error+=sum_error
            error=error.cpu().data.numpy()/(length)
            acc_train=str(100-error*100)+'%'
            print('train acc:'+acc_train)


       
torch.save(rnn,MOD)     
# for i, (x, y) in enumerate(test_loader):
#     sum_error=loss_fun(torch.zeros(len(test_data),3),y.float())
#     error=sum_error.cpu().data.numpy()/len(test_loader)
#     print('with all to be zero, average error:'+str(error))

