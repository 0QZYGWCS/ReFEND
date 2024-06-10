import pickle
import torch
import dgl
from settings import args
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from settings import MAX_COMMENT_COUNT

class MyDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = len(data)
        
    def __getitem__(self, mask):
        label = self.label[mask]
        data = self.data[mask]
        return data,label

    def __len__(self):
        return self.length

def con_loader_train_test(train_data_list,train_label_list,test_data_list,test_label_list):   
    train=MyDataSet(train_data_list,train_label_list)
    test=MyDataSet(test_data_list,test_label_list)
   
    train_loader = DataLoader(train, batch_size=args.Batch_size, shuffle=True,
                               collate_fn=collate_M)   
    test_loader=DataLoader(test,batch_size=args.Batch_size, shuffle=True, collate_fn=collate_M)
  
    return train_loader,test_loader

def collate_M(samples):
    graphs, labels = map(list, zip(*samples)) 
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

def graph(list_of_stance,list_1,comment_feature,news_feature):   
    
    u1=[];v1=[];u2=[];v2=[];u3=[];v3=[]
    T_all=zip(list_of_stance,list_1)

    favor=[];neutral=[];negative=[];
    ##three category
    for i,j in T_all:
        if i==2:
            favor.append(j)
        if i==1:
            neutral.append(j)
        if i==0:
            negative.append(j)
  
    for i in range(len(favor)):
        u1.extend([i for j in range(len(favor)-i)])
        for j in range(len(favor)):
            if j>=i:
                    v1.append(j)
   
    for i in range(len(neutral)):
        u2.extend([i for j in range(len(neutral)-i)])
        for j in range(len(neutral)):
            if j>=i:
                    v2.append(j)
    
    for i in range(len(negative)):
        u3.extend([i for j in range(len(negative)-i)])
        for j in range(len(negative)):
            if j>=i:
                    v3.append(j)
    
    graph_data={

    ('fcomment','favor','content'):(torch.IntTensor([i for i in range(len(favor))]),torch.IntTensor([0 for x in range(len(favor))])),
    ('content','favor','fcomment'):(torch.IntTensor([0 for x in range(len(favor))]),torch.IntTensor([i for i in range(len(favor))])),
    
    ('ncomment','neutral','content'):(torch.IntTensor([i for i in range(len(neutral))]),torch.IntTensor([0 for x in range(len(neutral))])),
    ('content','neutral','ncomment'):(torch.IntTensor([0 for x in range(len(neutral))]),torch.IntTensor([i for i in range(len(neutral))])),

    ('negcomment','negative','content'):(torch.IntTensor([i for i in range(len(negative))]),torch.IntTensor([0 for x in range(len(negative))])),
    ('content','negative','negcomment'):(torch.IntTensor([0 for x in range(len(negative))]),torch.IntTensor([i for i in range(len(negative))])),

    ('ncomment','favor','ncomment'):(torch.IntTensor(u2),torch.IntTensor(v2)),
    ('ncomment','favor','ncomment'):(torch.IntTensor(v2),torch.IntTensor(u2)),

    ('fcomment','favor','fcomment'):(torch.IntTensor(u1),torch.IntTensor(v1)),
    ('fcomment','favor','fcomment'):(torch.IntTensor(v1),torch.IntTensor(u1)),

    ('negcomment','favor','negcomment'):(torch.IntTensor(u3),torch.IntTensor(v3)),
    ('negcomment','favor','negcomment'):(torch.IntTensor(v3),torch.IntTensor(u3)),         
    }
    g=dgl.heterograph(graph_data)
    if len(neutral)!=0:
        g.nodes['ncomment'].data['feat']=torch.cat(([comment_feature[j].unsqueeze(0) for j in neutral]),0)
    if len(favor)!=0:
        g.nodes['fcomment'].data['feat']=torch.cat(([comment_feature[i].unsqueeze(0) for i in favor]),0)
    if len(negative)!=0:
        g.nodes['negcomment'].data['feat']=torch.cat(([comment_feature[m].unsqueeze(0) for m in negative]),0)
    g.nodes['content'].data['feat']=news_feature        

    return g 

    

#####################################################################################################
def weibo20_train_test(path):
    with open(path,"rb") as f:
        newsdic = pickle.load(f)
    print("newsdic load successfully!")
    true_list = []
    fake_list = []
    for id ,newsinfo in newsdic.items():
        if(newsinfo["lable"]==1): fake_list.append(id)
        elif(newsinfo["lable"]==0): true_list.append(id)
    print(f"len of newsdic: {len(newsdic)},fake_list: {len(fake_list)},real_list: {len(true_list)}")
    
    #divide train and test set randomly
    test_ratio = 0.25
    
    number_1 = int(test_ratio*len(fake_list)) 
    number_2 = int(test_ratio*len(true_list))  
    import random
    sample1 = random.sample(fake_list, number_1)
    sample2 = random.sample(true_list, number_2)
    test_id_list = sample1+sample2
    train_id_list = [id for id in list(newsdic.keys()) if id not in test_id_list]
    print("train:{},test:{}".format(len(train_id_list),len(test_id_list)))
    
    train_data_list = []
    test_data_list = []
    train_label_list = []
    test_label_list = []
 
    
    for id,newsinfo in newsdic.items():
        list_of_sentiment=newsinfo["sentiments"]
        if(len(list_of_sentiment)>MAX_COMMENT_COUNT):
            list_of_sentiment = newsinfo["sentiments"][:MAX_COMMENT_COUNT]
        A=[x for x in range(0,len(list_of_sentiment))]
        comments_feature = torch.tensor(newsdic[id]["comments_feature"])   
        news_feature = torch.tensor(newsdic[id]["news_feature"])   
       # print(comments_feature.shape,news_feature.shape)
        
        g = graph(list_of_stance=list_of_sentiment,list_1=A,news_feature=news_feature,comment_feature=comments_feature)
        label=int(newsinfo["lable"])
        
        if id in train_id_list:
            train_data_list.append(g)
            train_label_list.append(label)
        elif id in test_id_list:
            test_data_list.append(g)
            test_label_list.append(label)
            
    print("dataloader buliding...")
    train_loader,test_loader=con_loader_train_test(train_data_list,train_label_list,test_data_list,test_label_list)
    return train_loader,test_loader


def weibocomp_train_test(path):
    with open(path,"rb") as f:
        newsdic = pickle.load(f)
    print("newsdic load successfully!")
   
    true_list = []
    fake_list = []
    for id ,newsinfo in newsdic.items():
        if(newsinfo["lable"]==1): fake_list.append(id)
        elif(newsinfo["lable"]==0): true_list.append(id)
    print(f"len of newsdic: {len(newsdic)},fake_list: {len(fake_list)},real_list: {len(true_list)}")
  
    test_ratio = 0.25
    
    number_1 = int(test_ratio*len(fake_list)) 
    number_2 = int(test_ratio*len(true_list))  
    import random
    sample1 = random.sample(fake_list, number_1)
    sample2 = random.sample(true_list, number_2)
    test_id_list = sample1+sample2
    train_id_list = [id for id in list(newsdic.keys()) if id not in test_id_list]
    
    print("train:{},test:{}".format(len(train_id_list),len(test_id_list)))
    
        
    train_data_list = []
    test_data_list = []
    
    train_label_list = []
    test_label_list = []
 
    for id,newsinfo in newsdic.items():
        list_of_sentiment=newsinfo["sentiments"]
            
        if(len(list_of_sentiment)>MAX_COMMENT_COUNT):
            list_of_sentiment = newsinfo["sentiments"][:MAX_COMMENT_COUNT]
       
        A=[x for x in range(0,len(list_of_sentiment))]
        comments_feature = torch.tensor(newsdic[id]["comments_feature"])   
        news_feature = torch.tensor(newsdic[id]["news_feature"])  
        #print(comments_feature.shape,news_feature.shape)
        g = graph(list_of_stance=list_of_sentiment,list_1=A,news_feature=news_feature,comment_feature=comments_feature)
        label=int(newsinfo["lable"])
        
        if id in train_id_list:
            train_data_list.append(g)
            train_label_list.append(label)
        elif id in test_id_list:
            test_data_list.append(g)
            test_label_list.append(label)
    
    print("dataloader buliding...")
    train_loader,test_loader=con_loader_train_test(train_data_list,train_label_list,test_data_list,test_label_list)
    return train_loader,test_loader


def politifact_train_test(path):
    with open(path,"rb") as f:
        newsdic = pickle.load(f)
    print("newsdic load successfully!")
    true_list = []
    fake_list = []
  
    for id ,newsinfo in newsdic.items():
        if(newsinfo["lable"]==1): fake_list.append(id)
        elif(newsinfo["lable"]==0): true_list.append(id)
    print(f"len of newsdic: {len(newsdic)},fake_list: {len(fake_list)},real_list: {len(true_list)}")
  
    test_ratio = 0.25
    
    number_1 = int(test_ratio*len(fake_list)) 
    number_2 = int(test_ratio*len(true_list))  
    
    import random
    sample1 = random.sample(fake_list, number_1)
    sample2 = random.sample(true_list, number_2)
    test_id_list = sample1+sample2
    train_id_list = [id for id in list(newsdic.keys()) if id not in test_id_list]
    print("train: {},test: {}".format(len(train_id_list),len(test_id_list)))
    
        
    train_data_list = []
    test_data_list = []
    
    train_label_list = []
    test_label_list = []
 
    for id,newsinfo in newsdic.items():
        list_of_sentiment=newsinfo["sentiments"]
        if(len(list_of_sentiment)>MAX_COMMENT_COUNT):
            list_of_sentiment = newsinfo["sentiments"][:MAX_COMMENT_COUNT]
        
        A=[x for x in range(0,len(list_of_sentiment))]
        comments_feature = torch.tensor(newsdic[id]["comments_feature"]) 
        news_feature = torch.tensor(newsdic[id]["news_feature"])   
        #print(comments_feature.shape,news_feature.shape)
        g = graph(list_of_stance=list_of_sentiment,list_1=A,news_feature=news_feature,comment_feature=comments_feature)

        label=int(newsinfo["lable"])
        
        if id in train_id_list:
            train_data_list.append(g)
            train_label_list.append(label)
        elif id in test_id_list:
            test_data_list.append(g)
            test_label_list.append(label)
    
    print("dataloader buliding...")
    train_loader,test_loader=con_loader_train_test(train_data_list,train_label_list,test_data_list,test_label_list)
    return train_loader,test_loader

##########################################################
path = f"dataset/{args.dataset}/All_emb_sen.pickle"
print("*"*20,"loading data...","*"*20)
if args.dataset=="Weibo20":
    train_loader,test_loader= weibo20_train_test(path)
elif args.dataset=="Weibo-comp":
    train_loader,test_loader = weibocomp_train_test(path)
elif args.dataset =="PolitiFact":
    train_loader,test_loader = politifact_train_test(path)    
