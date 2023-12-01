import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch
from data_load import preDataset
from sklearn import metrics
from dgl.nn.pytorch import GraphConv
import os
import numpy as np
from settings import model_param_save,EMBEDDING_DIMENSION,LEARNING_RATE,EPOCH

mpl.use('Agg')


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)   
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)  
            for rel in rel_names}, aggregate='sum')
    def forward(self, graph, inputs):     
        h = self.conv1(graph, inputs)  
        h = {k: F.relu(v) for k, v in h.items()}  
        h = self.conv2(graph, h)   
        h = {k: F.relu(v) for k, v in h.items()}
        return h

class HeteroClassifier(nn.Module):  
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names) 
        self.classify = nn.Linear(hidden_dim, n_classes)  

    def forward(self, g):
        h = g.ndata['feat'] 
        h = self.rgcn(g, h)  
        with g.local_scope():  
            g.ndata['h'] = h
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)   
            return self.classify(hg)

def train(model,data_loader,opt):
    model.train()
    loss_total=0;
    for batched_graph, labels in data_loader:
        batched_graph=batched_graph.to('cuda:1')
        labels=labels.to('cuda:1')
        logits = model(batched_graph)  
        loss = F.cross_entropy(logits, labels.squeeze(-1))
        opt.zero_grad()    
        loss.backward()
        opt.step()          
        loss_total+=loss
    loss_total_numpy=loss_total.cpu().detach().numpy()
    return loss_total_numpy

@torch.no_grad()
def test(model,test_loader,flag):
    model.eval()   
    list_predicted=[]
    num_correct=0;
    num_total=0;
    list_true=[];
    loss_total=0;
    for batched_graph,labels in test_loader:
        batched_graph=batched_graph.to('cuda:1')
        labels=labels.to('cuda:1')
        logits=model(batched_graph)
        predict=logits.argmax(1)
        num_total+=len(labels)
        num_correct+=(predict==labels).sum()
        loss=F.cross_entropy(logits, labels.squeeze(-1))
        loss_total+=loss.cpu().detach().numpy()
        list_predicted.extend(predict.to('cpu'))
        list_true.extend(labels.to('cpu'))   
        
    if flag=='train':
        return metrics.accuracy_score(list_true, list_predicted)
    if flag=='test':
        print("********************Result********************")
        print("Number of news for test{}，ACC:".format(num_total),metrics.accuracy_score(list_true, list_predicted))
        print("precision_score:", metrics.precision_score(list_true, list_predicted))
        print("recall_score:", metrics.recall_score(list_true, list_predicted))
        print("f1_score:", metrics.f1_score(list_true, list_predicted))
            
def run_code():
  
    dataset=preDataset()  
    dataset.construct_newwdic()  #
    print("graph data construct")
    dataset.construct_graph()
    print("dataset split")
    data_loader,test_loader=dataset.divide_train_test()
    etypes = data_loader.dataset[0][0].etypes      
    model = HeteroClassifier(768,EMBEDDING_DIMENSION, 2,etypes)  
    model.to('cuda:1')
    opt = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=5e-3)    
    print("Training, please wait a while...")
   
   
    for i in range(EPOCH):
        loss=train(model=model,data_loader=data_loader,opt=opt)
        train_acc=test(model=model,test_loader=data_loader,flag='train')
        print("epcoh,{}".format(i),"train_loss:{}".format(loss),"train_acc:{}".format(train_acc))
  
    torch.save(model.state_dict(), model_param_save)   
    print("Model parameters saved successfully.")
    print("Testing...")
    test(model=model,test_loader=test_loader,flag='test')
    print("Finished!")
  
if __name__ == '__main__':
    run_code()

