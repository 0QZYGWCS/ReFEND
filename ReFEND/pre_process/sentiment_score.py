#
import sys
sys.path.append('../')
from snownlp import SnowNLP
from tqdm import tqdm
import csv
import numpy as np
import torch
import os
import pickle
from settings import MAX_COMMENT_COUNT
#######################################################for Chinese
def sentiment_chinese():
    filename_list=["Weibo20","Weibo-comp"]
    for name in filename_list:
        print(name)
        path = f"../dataset/{name}/{name}.pickle"
        save_path = f"../dataset/{name}/sentiment"
        tf=open(path,"rb")
        newsdic=pickle.load(tf)
    
     
        print("Calculating sentiment score of comments")
    
        for newid,newsitem in tqdm(newsdic.items()):
            list_sentiments = []
            list_of_comments=newsdic[newid]['comments']
            if(len(list_of_comments)>200):list_of_comments=list_of_comments[:200]
            for sentence in list_of_comments:
                s=SnowNLP(sentence)
                score=s.sentiments
                list_sentiments.append(score)
            newsitem["sentiments"] = list_sentiments
        dic_path = os.path.join(f"../dataset/{name}/","sentiment_dic_"+name+".pickle")
        with open(dic_path,'wb') as f:
            pickle.dump(newsdic,f)
        with open(dic_path,'rb') as f:
            dic = pickle.load(f)    
            print(len(dic))
    
##########################################for English
def sentiment_english():
    import sys
    sys.path.append('./')
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import csv
    import pickle
    name = "PolitiFact"
    
    analyzer = SentimentIntensityAnalyzer()
    with open(f"../dataset//{name}/{name}.pickle",'rb') as f:
        newsdic=pickle.load(f)
    for newsid,newsitem in newsdic.items():
        list_comments=newsdic[newsid]['comments']
        if(len(list_comments)>MAX_COMMENT_COUNT): list_comments = list_comments[:MAX_COMMENT_COUNT]
        list_sentiment=[]
        for sentence in list_comments:
            vs= analyzer.polarity_scores(sentence)
            list_sentiment.append(vs['compound'])
        newsitem["sentiments"] = list_sentiment
    
    with open(f"../dataset/{name}/sentiment_dic_{name}.pickle",'wb') as f:
        pickle.dump(newsdic,f)
        
print("Processing Chinese dataset")
sentiment_chinese()
print("Processing English dataset")
sentiment_english()
