# 文档理解 -- LayoutLMV1

### 数据集选用的是SROIE，参考了kaggle上面的实现，主要记录与备份lmv1从数据集到测试的相关代码

### 数据集准备
```
https://drive.google.com/file/d/1cWjCGrmEmk0tcReyC7WDiJXkjzk2NM6a/view?usp=sharing
```
解压上面数据集与BERT，与preprocess_data.py并列
```
执行 python preprocess_data.py
```
#### 训练与测试
执行 `unilm/layoutlm/deprecated/examples/seq_labeling/train.sh`
脚本自动执行训练与测试，预测结果保存在train.log中