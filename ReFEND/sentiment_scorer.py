'''
Before use snownlp to evaluate the comment sentiment,
please pretrain the model through  Weibo sentiment corpus (https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets
/weibo_senti_100k/intro.ipynb) and refer to the official documentation of Snownlp for specific implementation.
'''

'''
Sentiment scoring 
'''
import sys
sys.path.append('./')
from snownlp import SnowNLP
import csv
import json
import os
from REFEND_settings import dataset_path,sentiment_file
def cal_comments_score():
    tf=open(dataset_path,"r")
    newsdic=json.load(tf)
    outputfile=open(sentiment_file,encoding='utf-8',mode='w',newline='')
    csv_wirter=csv.writer(outputfile)
    print("Sentiment scoring is processing.")
    for newid,newsitem in newsdic.items():
        list_of_comments=newsdic[newid]['comments']
        csv_comments=[];csv_score=[];
        for sentence in list_of_comments:
            s=SnowNLP(sentence)
            score=s.sentiments
            csv_comments.append(sentence)
            csv_score.append(score)
        csv_rows=zip(csv_comments,csv_score)
        csv_wirter.writerows(csv_rows)
    outputfile.close()

def main():
    cal_comments_score()
    print("Sentiment scoring has been fisished.")

if __name__=='__main__':
    main()