
import pickle
def read(name):
    print("loading embedding...")
    with open(f"../dataset/{name}/embedding.pickle",'rb') as f:
        text_feature = pickle.load(f)  #key: ['comments_feature', 'news_feature']
    # print({key:text_feature[key] for key,item in list(text_feature.items())[:1]})
    print("loading data with sentiments")
    with open(f"../dataset/{name}/sentiment_dic_{name}.pickle",'rb') as f:
        sentiment = pickle.load(f)
    #print({key:sentiment[key] for key,item in list(sentiment.items())[:1]})
    ###show keys of newsdic 
    print([list(newsinfo.keys()) for id ,newsinfo in list(text_feature.items())[:1]]) #['comments_feature', 'news_feature']]

    for id,info in sentiment.items():
        data = info["sentiments"]
        #print(data)
        k = comp(name,data)
        info["sentiments"] = k
    print([list(newsinfo.keys()) for id ,newsinfo in list(sentiment.items())[:1]]) #['news_content', 'comments', 'lable', 'sentiments']
    ###integrate##### 
    for id,newsinfo in sentiment.items():
        newsinfo["news_feature"] = text_feature[id]["news_feature"]
        newsinfo["comments_feature"] = text_feature[id]["comments_feature"]
    
    print([list(newsinfo.keys()) for id,newsinfo in list(sentiment.items())[:1]])   #['news_content', 'comments', 'lable', 'sentiments', 'news_feature', 'comments_feature']
 
    ##save integrate dic which consist of news,comments,text_embedding and sentiments
    text_feature = None
    with open(f"../dataset/{name}/All_emb_sen.pickle",'wb') as f:
        pickle.dump(sentiment,f)
    print("The merging process has been completed.")
    
def comp(name,data):
    u_data = [-1]*len(data)
    ##three categories, make mapping: 2:favor, 1:neutral 0:negative
    if name=="Weibo20" or name=="Weibo-comp":
        intervals = [(0, 1/3), (1/3, 2/3), (2/3, 1)]  
    elif name=="PolitiFact":
        intervals = [(-1, -1/3), (-1/3, 1/3), (1/3, 1)]
    for i in range(len(data)):
        for j in range(len(intervals)):
            if intervals[j][0] <= data[i] <intervals[j][1]:
                u_data[i] =j
                break
    return u_data
    
if __name__ =="__main__":
    dataset = ["Weibo20","Weibo-comp","PolitiFact"]
    for name in dataset:
        print("*"*20,"Processing {} now".format(name),"*"*20)
        read(name)
