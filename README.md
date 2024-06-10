# ReFEND
This work is being submitted to the TKDE, and it will be gradually improved based on the feedback.
### Raw_Dataset

|Dataset|Resource|
|--------|--------|
|Weibo20|[https://github.com/fip-lab/STANKER/tree/main/raw_datasets](https://github.com/fip-lab/STANKER/tree/main/raw_datasets)|
|Weibo-comp|[https://www.datafountain.cn/competitions/422](https://www.datafountain.cn/competitions/422)|
|PolitiFact|[https://github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)|

You can customize the dataset to your needs. For this work, we processed this dataset and provided the dataset file, you can download it from 
this link:：https://pan.baidu.com/s/1ujpy-lFPKFN2snpGz58Tbw      **code**：**mvtm**

After downloading the dataset, please ensure to place it in the appropriate folder as demonstrated below:
```
-dataset
   --Weibo20
     ---Weibo20.pickle (processed)
     ---Weibo20.json (raw file)
```
the content format of the pickle file is :

```
{
  news_id:{
               "news_content": "xxxx"
               "comments":["xxx","xxx","xxx"]
               "label": 0 
          }
  ......
}
```

### Code
**Requirements**:
```
python=3.7.0
dgl-cu101=0.6.1
torch=1.8.1+cu101
torchaudio=0.8.1
torchvision=0.9.1+cu101
pytorch-transformers=1.2.0
snownlp=0.12.3
```

**step1**: Text Encoding
```
cd pre_process
python text_encoding.py
```
After that, the embedding will  be generated in the corresponding folder, for example, for Weibo-comp, the embeddings will be stored in *“dataset/Weibo-comp/embedding.pickle”*

**step2**: Sentiment score
```
cd pre_process
python sentiment_scorer.py
```
After that, the sentiment score will be generated, and *‘sentiment_dic_{dataset_name}.pickle’* is created. For Weibo-comp, the path is *"dataset/Weibo-comp/sentiment_dic_Weibo-comp.pickle"*

**step3**: Integrate text encoding and sentiment score into newsdic
```
cd pre_process
python cat.py
```
Then, *‘All_emb_sen.pickle’* will be generated in the corresponding folder.
After preprocessing work has been completed, the contents of the folder are as follows (Weibo-comp dataset as example):

```
- dataset
  - Weibo-comp
    - embedding.pickle
    - sentiment_dic_Weibo-comp.pickle
    - All_emb_sen.pickle
    - Weibo-comp.pickle
```


**step4**: Run ReFEND to get results 
```
cd ..
python run.py 
```
After that, the model parameters will be stored in the corresponding folder *"param/{dataset_name}/"*.  The result will be stored in *"result/{dataset_name}.txt"* 

Please indicate the source if you need to use the work in this article. Thank you.






