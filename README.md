# ReFEND
This work is being submitted to the WWW conference (2024).
### Dataset

|Dataset|Resource|
|--------|--------|
|Weibo16|[https://github.com/RMSnow/WWW2021](https://github.com/RMSnow/WWW2021)|
|Weibo20|[https://github.com/RMSnow/WWW2021](https://github.com/RMSnow/WWW2021)|
|Weibo-comp|[https://www.datafountain.cn/competitions/422](https://www.datafountain.cn/competitions/422)|
|Politifact|[https://github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)|

After downloading the dataset, place it in the corresponding folder, as shown below.
- weibo16
  - comments_embedding
  - model_param
  - news_embedding
  - **weibo16_dict.json**

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
python text_encoding.py
```
**step2**: Sentiment score
```
python sentiment_scorer.py
```


**step3**: Run ReFEND to get results 
```
run_model.py 
```
After that, the model parameters will be stored in the corresponding folder, for example, weibo16/model_param.

If you need to use the work in this article, please indicate the source. Thank you.





