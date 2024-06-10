import pickle
import sys
sys.path.append('../')

def get_embedding(name):
    path = f"../dataset/{name}/{name}.pickle"
    with open(path,"rb") as f:
        newsdic = pickle.load(f)
    import torch
    from transformers import BertTokenizer,BertModel
    import re
    import numpy
    from settings import MAX_CONTENT_LENGTH,MAX_COMMENT_COUNT
    #loading tokenizer and model
    if name =="PolitiFact":  #for English
        tokenizer  = BertTokenizer.from_pretrained("bert-base-uncase")
        model = BertModel.from_pretrained("bert-base-uncase")
    elif name=="Weibo20" or name=="Weibo-comp":  #for Chinese
        tokenizer  = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
    print("tokenzier and model have benn load successfully")
    text_feature = {}
    from tqdm import tqdm
    print("Getting embeding... This Process need some time...")
    with torch.no_grad():
        for newsid,newsinfo in tqdm(newsdic.items(),total = len(newsdic)):  
            comments_list= newsinfo["comments"][:MAX_COMMENT_COUNT]
            news_feature = []
            comments_feature =[]
            text_feature[newsid] = {}
            ## get comments embedding
            for comments in comments_list:
                tok = tokenizer.encode(comments)
                if(len(tok)>=MAX_CONTENT_LENGTH):
                    comments_input_id = torch.tensor(tok[0:MAX_CONTENT_LENGTH]).unsqueeze(0)
                else:
                    comments_input_id = torch.tensor(tok).unsqueeze(0)
                comment_output = model(comments_input_id)    
                comments_feature.extend(numpy.array(comment_output[1].detach().numpy()).tolist())
            # print("comments embedding shape：",numpy.array(comments_feature).shape)  
            text_feature[newsid]['comments_feature'] = comments_feature
         
            ## get news embedding
            news = newsdic[newsid]['news_content']
            tok = tokenizer.encode(news)
            if(len(tok)>=MAX_CONTENT_LENGTH):
                news_input_id = torch.tensor(tok[0:MAX_CONTENT_LENGTH]).unsqueeze(0)
            else:
                news_input_id = torch.tensor(tok).unsqueeze(0)
            news_output = model(news_input_id)
            news_feature.extend(numpy.array(news_output[1].detach().numpy()).tolist())
            # print("news embedding shape：",numpy.array(news_feature).shape)  
            text_feature[newsid]['news_feature'] = news_feature
            
        print("Now,saving embedding")
        path_save = f"../dataset/{name}/embedding.pickle"
        with open(path_save,'wb') as f:
            pickle.dump(text_feature,f)
        with open(path_save, 'rb') as f:
            text_feature = pickle.load(f)
        print("All embedding work have benn finished!")  
           
if __name__ =="__main__":
    dataset = ["Weibo-comp","Weibo20","PolitiFact"]
    for name in dataset:
        get_embedding(name)
