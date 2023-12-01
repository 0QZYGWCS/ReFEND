import csv
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
import torch
import dgl
from torch.utils.data import DataLoader
import json
import numpy as np
import os
from tqdm import tqdm 
from REFEND_settings import dataset_path,news_embedding_path,comment_emebedding_path,sentiment_file
from REFEND_settings import MIN_COMMENT_COUNT,MAX_COMMENT_COUNT,MIN_COMMENT_LENGTH,BATCH_SIZE,TEST_SIZE


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

class preDataset():
    def __init__(self):
        self.newsdic={}
        self.list_graph=[]
        self.list_label=[]
        comment_feature_dict_1={}
        comment_feature_dict_2={}
        print("Reading the embedding of comments, please wait a while...")
        for home, dirs, files in os.walk(comment_emebedding_path):
            for filename in tqdm(files):
                tf1 = open(os.path.join(home, filename), "r")
                comment_feature_dict_2 = json.load(tf1)
                comment_feature_dict_1.update(comment_feature_dict_2)
        print("Reading comments' embedding successfully!")
        for comment_id in comment_feature_dict_1:
            comment_feature_dict_1[comment_id] = torch.tensor(np.array(comment_feature_dict_1[comment_id])).to(torch.float32)    #一条评论768维度 
        print("Reading the embedding of news, please wait a while...")
        tf2=open(news_embedding_path,'r')
        news_feature_dict =json.load(tf2)
        for news_id in news_feature_dict:
            news_feature_dict[news_id]=torch.tensor(np.array(news_feature_dict[news_id])).to(torch.float32)
        self.comment_feature_dict=comment_feature_dict_1
        self.news_feature_dict=news_feature_dict
        print("Reading news embedding successfully!")
       

    def construct_newwdic(self):
        tf_data=open(dataset_path,"r")
        self.newsdic=json.load(tf_data)
        list_stance=[]
        # open the sentiment file to get score.
        with open(sentiment_file) as tsvfile:
            reader=csv.reader(tsvfile)
            for row in reader:
                stance=float(row[1])
                if 1/3<= stance <2/3:
                    stance=0.5;
                elif stance<1/3:stance=0;
                else:stance=1;
                list_stance.append(stance) 
        index_start=0
        index_end=0
        for news_id,news_info in self.newsdic.items():     
            len_of_comments=len(self.newsdic[news_id]['comments'])
            index_end=len_of_comments+index_end
            self.newsdic[news_id]['stance']=list_stance[index_start:index_end]
            index_start=index_start+len_of_comments

        count_after=0;count_after_news=0;count_label0=0;count_label1=0
        for news_id ,news_info in self.newsdic.items():
            list_1=[];list_2=[];
            list_of_stance=self.newsdic[news_id]['stance']
            list_Of_comments=self.newsdic[news_id]['comments']   
            
            if len(list_Of_comments)<MIN_COMMENT_COUNT:
                continue;
            del_index=[]
            for index_comment,comment_text in enumerate(list_Of_comments):
                if len(comment_text)>=MIN_COMMENT_LENGTH:
                    list_1.append(index_comment)
                    list_2.append(comment_text)
                else:
                    del_index.append(index_comment)   
            if len(list_1)<MIN_COMMENT_COUNT :continue;
            list_of_stance=[list_of_stance[i] for i in range(len(list_of_stance)) if (i not in del_index)]   
            if len(list_1)>=MAX_COMMENT_COUNT:    
                list_of_stance=list_of_stance[:MAX_COMMENT_COUNT]
            count_after=count_after+len(list_of_stance)
            count_after_news=count_after_news+1
            
            if self.newsdic[news_id]['label']==1:
                count_label1=count_label1+1
            else:
                count_label0=count_label0+1
        
        print("*****************dataset*********************")
        print("The number of news:",count_after_news)
        print("The number of comments:",count_after)  
        print("The number of news(labeled 1)",count_label1)
        print("The number of news(labeled 0)",count_label0)
    
    
    def graph(self,list_of_stance,list_1,news_id): 
    
        u=[];v=[];u2=[];v2=[];u3=[];v3=[]
        T_all=zip(list_of_stance,list_1)
    
        favor=[];neutral=[];against=[];
        for i,j in T_all:
            if i==1:
                favor.append(j)
            if i==0.5:
                neutral.append(j)
            if i==0:
                against.append(j)
       
        for i in range(len(favor)):
            u.extend([i for j in range(len(favor)-i)])
            for j in range(len(favor)):
                if j>=i:
                     v.append(j)
        
        for i in range(len(neutral)):
            u2.extend([i for j in range(len(neutral)-i)])
            for j in range(len(neutral)):
                if j>=i:
                     v2.append(j)
       
        for i in range(len(against)):
            u3.extend([i for j in range(len(against)-i)])
            for j in range(len(against)):
                if j>=i:
                     v3.append(j)
        graph_data={

        ('fcomment','favor','content'):(torch.IntTensor([i for i in range(len(favor))]),torch.IntTensor([0 for x in range(0,len(favor))])),
        ('content','favor','fcomment'):(torch.IntTensor([0 for x in range(0,len(favor))]),torch.IntTensor([i for i in range(len(favor))])),
        
        ('ncomment','negative','content'):(torch.IntTensor([i for i in range(len(neutral))]),torch.IntTensor([0 for x in range(0,len(neutral))])),
        ('content','negative','ncomment'):(torch.IntTensor([0 for x in range(0,len(neutral))]),torch.IntTensor([i for i in range(len(neutral))])),

        ('acomment','neutral','content'):(torch.IntTensor([i for i in range(len(against))]),torch.IntTensor([0 for x in range(0,len(against))])),
        ('content','neutral','acomment'):(torch.IntTensor([0 for x in range(0,len(against))]),torch.IntTensor([i for i in range(len(against))])),

        ('ncomment','favor','ncomment'):(torch.IntTensor(u2),torch.IntTensor(v2)),
        ('ncomment','favor','ncomment'):(torch.IntTensor(v2),torch.IntTensor(u2)),

        ('fcomment','favor','fcomment'):(torch.IntTensor(u),torch.IntTensor(v)),
        ('fcomment','favor','fcomment'):(torch.IntTensor(v),torch.IntTensor(u)),
    
        ('acomment','favor','acomment'):(torch.IntTensor(u3),torch.IntTensor(v3)),
        ('acomment','favor','acomment'):(torch.IntTensor(v3),torch.IntTensor(u3)),         
        }
        g=dgl.heterograph(graph_data)
        if len(neutral)!=0:
            g.nodes['ncomment'].data['feat']=torch.cat(([self.comment_feature_dict[news_id][j].unsqueeze(0) for j in neutral]),0)
        if len(favor)!=0:
            g.nodes['fcomment'].data['feat']=torch.cat(([self.comment_feature_dict[news_id][i].unsqueeze(0) for i in favor]),0)
        if len(against)!=0:
            g.nodes['acomment'].data['feat']=torch.cat(([self.comment_feature_dict[news_id][m].unsqueeze(0) for m in against]),0)
        g.nodes['content'].data['feat']=self.news_feature_dict[news_id]        

        return g
       
       
    def construct_graph(self):
        for news_id ,news_info in self.newsdic.items():
            list_1=[];list_2=[];
            list_Of_comments=[];
            list_Of_content=[];list_of_stance=[]
            list_Of_comments=self.newsdic[news_id]['comments']
            list_of_stance=self.newsdic[news_id]['stance']
            list_Of_content.append(self.newsdic[news_id]['news_content'])
            # Filtering 
            if (len(list_Of_comments)<MIN_COMMENT_COUNT):
                continue;
            del_index=[]
            for index_comment,comment_text in enumerate(list_Of_comments):
                if len(comment_text)>=MIN_COMMENT_LENGTH:
                    list_1.append(index_comment)
                    list_2.append(comment_text)
                else:
                    del_index.append(index_comment)
            if len(list_1)<MIN_COMMENT_COUNT :continue;
            news_list_of_stance=[list_of_stance[i] for i in range(len(list_of_stance)) if (i not in del_index)]   
            if len(list_1)>=MAX_COMMENT_COUNT:    
                news_list_of_stance=news_list_of_stance[:200]
            A=[x for x in range(0,len(news_list_of_stance))] 
            g=self.graph(news_list_of_stance,A,news_id) 
            self.list_label.append(int(self.newsdic[news_id]['label']))     
            self.list_graph.append(g)  
            print("All graphs have been constructed.")

    def collate_M(self,samples):
        graphs, labels = map(list, zip(*samples)) 
        return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)
    
    def divide_train_test(self):
        train_set_list_data=[];train_set_list_label=[]
        test_set_list_data=[];test_set_list_label=[]
        data=list(zip(self.list_graph,self.list_label))
        train_set,test_set=train_test_split(data,test_size=TEST_SIZE,random_state=42)
        train_set_list_data[:],train_set_list_label[:]=zip(*train_set)
        test_set_list_data[:],test_set_list_label[:]=zip(*test_set)
         
        MyDataSet_1=MyDataSet(train_set_list_data,train_set_list_label)
        MyTestSet=MyDataSet(test_set_list_data,test_set_list_label)
        
        data_loader = DataLoader(MyDataSet_1, batch_size=BATCH_SIZE, shuffle=True,
                         collate_fn=self.collate_M)   
        test_loader=DataLoader(MyTestSet,batch_size=BATCH_SIZE, shuffle=True,
                         collate_fn=self.collate_M)
        return data_loader,test_loader

