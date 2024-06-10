import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from model import HeteroClassifier
import torch.nn.functional as F
from settings import args
from data_load import train_loader,test_loader
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy
import os

if(torch.cuda.is_available()):
    device = "cuda"
else :
    device = "cpu"

def train(model,train_loader,args,test_loader): 
    optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate,weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_f1 = float(0.0)  
    best_state = None
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        for batch_data,labels in train_loader:  
            batch_data = batch_data.to(device)
            labels = labels.to(device)   
            if(len(labels)==1) :
                labels = labels
            else:
                labels =labels.squeeze(-1)
            optimizer.zero_grad()
            outputs= model(batch_data) 
            loss = criterion(outputs,labels)
            epoch_loss = epoch_loss+loss.item()
            loss.backward()
            optimizer.step()
        print("*"*80,f"Epoch{epoch+1}/{args.num_epochs},Loss:{epoch_loss}")
        print("Testing on test set ")
        a,b,c,d = test(model,test_loader)
        test_loss = calculate_test_loss(model,test_loader,criterion)
        print(f'Epoch {epoch+1}/{args.num_epochs},Loss:{epoch_loss},Test Loss:{test_loss}')
        if(d>best_f1):
            best_state = model.state_dict()
            best_f1 = d
            best_result = [a,b,c,d]
    print("stopping...")
    print("saving model parameters...")
 
    if not os.path.exists(f"param/{args.dataset}"):
        os.makedirs(f"param/{args.dataset}")
    torch.save(best_state,f'param/{args.dataset}/_save_model.pth')
    print("saving successfully!")
    return best_result
        
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
def test(model,test_loader):
    model.eval()
    y_true=[]
    y_pre = []
    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)  
            _, predicts = torch.max(outputs,1)  
            y_true.extend(labels.cpu().numpy())
            y_pre.extend(predicts.cpu().numpy())
            
            accuracy = accuracy_score(y_true,y_pre)
            precision = precision_score(y_true,y_pre)
            recall = recall_score(y_true,y_pre)
            f1 = f1_score(y_true,y_pre)
            print(accuracy,precision,recall,f1)
            return accuracy,precision,recall,f1
             
def calculate_test_loss(model,test_loader,criterion): 
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for data,labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs,labels)
            valid_loss +=loss.item()
        return valid_loss   
    
def save_result(a,b,c,d,flag="every"):
    print("saving results...")
    
    with open(f'result/{args.dataset}_result.txt','a+') as f:
        f.write(f"------------{flag}-------time = {args.time}------lr={args.learning_rate}------------\n")
        f.write(f'Acc:{a}')
        f.write(" "*10)
        f.write(f'Pre:{b}')
        f.write(" "*10)
        f.write(f'Recall:{c}')
        f.write(" "*10)
        f.write(f'F1:{d}\n')
        f.write("---------------------------------------------")
        f.close()

if __name__ == '__main__':
    result_array=[]
    for i in range(5):
        etypes = train_loader.dataset[0][0].etypes     
        model = HeteroClassifier(768,args.hidden_dim, args.class_num,etypes)  
        print("Begin Training...")
        model.to(device)
        a,b,c,d= train(model,train_loader,args,test_loader)
        save_result(a,b,c,d)
        result_array.append([a,b,c,d])
        if(i==4):
            a,b,c,d=numpy.array(result_array).mean(axis=0)
            save_result(a,b,c,d,"avg")
                
