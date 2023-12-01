'''
The file path that related to all project are listed as follow.
'''
import os
dataset = 'weibo16'
#Please download the weibo16 dataset and place it in the corresponding folder,
# like follows:
dataset_path = 'weibo16/weibo_dict_16.json'   

comment_emebedding_path = "weibo16/comments_embedding/"
news_embedding_path = 'weibo16/news_embedding/weibo16_news_feature_dict.json'
sentiment_file = 'weibo16/Weibo_Sentime.csv'
model_param_save = 'weibo16/model_param/params.pth'

'''
parameters are listed as follows:

'''

MIN_COMMENT_COUNT = 5
MAX_COMMENT_COUNT = 200
MIN_COMMENT_LENGTH = 2
BATCH_SIZE = 150
TEST_SIZE = 0.25
MAX_NEWS_LENGTH = 512
MAX_COMMENT_LENGTH = 512
EMBEDDING_DIMENSION = 256
LEARNING_RATE =  3e-5
EPOCH = 200