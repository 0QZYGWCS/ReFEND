import torch
import os
from REFEND_settings import dataset_path,news_embedding_path,comment_emebedding_path
from REFEND_settings import MIN_COMMENT_COUNT,MIN_COMMENT_LENGTH,MAX_NEWS_LENGTH,MAX_COMMENT_LENGTH 
#from pytorch_transformers import BertTokenizer,BertModel
from transformers import BertTokenizer, BertModel
import json
import numpy as np

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
print("success")
print(model)
model.eval()
print("the loading process of roberta is finished.")

from tqdm import tqdm
def generate_news_embedding():
    print("Text Encoding of news")
    news_feature_dict={}
    count=0
    tf=open(dataset_path,"r")
    newsdic=json.load(tf)
    with torch.no_grad():  
        for newsid,newsinfo in tqdm(newsdic.items(),total=len(newsdic)):
            count=count+1
            if(len(newsdic[newsid]['news_content'])>=MAX_NEWS_LENGTH):
                news_input_id =torch.tensor((tokenizer.encode(newsdic[newsid]['news_content']))[0:MAX_NEWS_LENGTH]).unsqueeze(0)
            else:
                news_input_id =torch.tensor(tokenizer.encode(newsdic[newsid]['news_content'])).unsqueeze(0)
            news_output=model(news_input_id)
            news_feature_dict[newsid]=np.array(news_output[1].detach().numpy()).tolist()    
        print("The number of new is :",count)  

    jsobj_news=json.dumps(news_feature_dict)
    fileobj_news=open(news_embedding_path,'w')
    fileobj_news.write(jsobj_news)
    fileobj_news.close()


def generate_comments_embedding():
    print("Encoding comments")
    count_comment=0;
    leng=0;
    comment_feature_dict={}
    count=0
    current_count=0;
    tf=open(dataset_path,"r")
    newsdic=json.load(tf)
    for id,news in newsdic.items():
        leng=leng+len(newsdic[id]['comments'])
    print("The number of comments is: {}".format(leng))
    total_i = len(newsdic)
    with torch.no_grad():  
        for newsid,newsinfo in tqdm(newsdic.items(), total=total_i):
            count=count+1;
            list_comments=[];
            list_comments=newsdic[newsid]['comments']  
            comment_vector=[]  
            if(len(list_comments)>0):   
                for sentences in list_comments:
                    if len(sentences)>=MIN_COMMENT_LENGTH:    
                        count_comment=count_comment+1;
                        current_count=current_count+1;
                        input_ids = torch.tensor(tokenizer.encode(sentences)).unsqueeze(0)  
                        outputs = model(input_ids)
                        comment_vector.append(outputs[1])
                if len(comment_vector)>0:   
                    comment_vector=torch.stack(comment_vector,dim=0).squeeze(1)
                    comment_feature_dict[newsid]= np.array(comment_vector.detach().numpy()).tolist() 
                    
            if current_count>80000 or count==len(newsdic):
                jsobj_comments=json.dumps(comment_feature_dict)
                fileobj_comments=open(os.path.join(comment_emebedding_path,'weibo16_comment_feature_dict_{count}.json'.format(count=count)),'w')
                fileobj_comments.write(jsobj_comments)
                fileobj_comments.close()
                comment_feature_dict={}
                current_count=0
        
def main():
    generate_news_embedding()
    print("New embedding is generated.")
    generate_comments_embedding()
    print("Comments embedding is generated.")
    print("=======Text Encoding have been finished.======")
    
if __name__ =='__main__':
    main()