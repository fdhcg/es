import torch 
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class RNN(nn.Module):
    def __init__(self,INPUT_SIZE,HIDDEN_SIZE,NUM_LAYER,BIDIRE,SL):
        super(RNN, self).__init__()
        self.conv1=nn.Conv1d(9,6,5,stride=1,padding=2)
        self.reluconv1=nn.ReLU()
        self.pool1=nn.MaxPool1d(5,stride=1,padding=2)
        self.conv2=nn.Conv1d(6,3,5,stride=1,padding=2)
        self.reluconv2=nn.ReLU()
        self.pool2=nn.MaxPool1d(5,stride=1,padding=2)
        self.conv3=nn.Conv1d(3,3,5,stride=1,padding=2)
        self.reluconv3=nn.ReLU()
        self.pool3=nn.MaxPool1d(5,stride=1,padding=2)
        self.conv4=nn.Conv1d(3,3,5,stride=1,padding=2)
        self.reluconv4=nn.ReLU()
        self.pool4=nn.MaxPool1d(5,stride=1,padding=2)
        self.fc=nn.Linear(3,3)
        self.rnn=nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYER,
            batch_first=True,
            bidirectional=BIDIRE,
            dropout=0.5
        )
        if BIDIRE:
            self.fc1=nn.Linear(HIDDEN_SIZE*2,HIDDEN_SIZE*2)
            self.out=nn.Linear(HIDDEN_SIZE*2,3)
            self.fc2=nn.Linear(HIDDEN_SIZE*2,HIDDEN_SIZE*2)

        else:
            self.fc1=nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
            self.fc2=nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
            self.out=nn.Linear(HIDDEN_SIZE,3)

        
        
        self.relu1=nn.ReLU()
        self.relu2=nn.ReLU()
 
        self.dropout1=nn.Dropout(0)
        self.dropout2=nn.Dropout(0)

        self.ems=nn.Linear(3,3)



    def forward(self, x, h_state):
        x1=self.conv1(x.permute(0,2,1))  #batch*in_channel*len
        x1=self.reluconv1(x1)
        x1=self.pool1(x1)
        x2=self.conv2(x1)
        x2=self.reluconv2(x2)
        x2=self.pool2(x2)
        x3=self.conv3(x2)
        x3=self.reluconv3(x3)
        x3=self.pool3(x3)
        x4=self.conv4(x3)
        x4=self.reluconv4(x4)
        x4=self.pool4(x4)
        x4=x4.permute(0,2,1)

        r_out, h_state = self.rnn(x, h_state)

       
        outs = []


        for time in range(r_out.size(0)): 
            fc1_out=self.fc1(r_out[time, :, :]) 
            relu1_out=self.relu1(fc1_out)
            dropout1_out=self.dropout1(relu1_out)
            fc2_out=self.fc2(dropout1_out)
            relu2_out=self.relu2(fc2_out)
            dropout2_out=self.dropout2(relu2_out)
            out=self.out(dropout2_out)    #120*3

            outs.append(self.ems(torch.add(out,self.fc(x4[time]))))

        return torch.stack(outs, dim=1)
        # return torch.stack(outs, dim=1), h_state


