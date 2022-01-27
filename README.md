# qianyan_similarity

## 1. 项目介绍
千言数据集：文本相似度比赛

比赛连接：https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition

## 2. 执行流程

```
数据准备:
PYTHONPATH=./ python data_process/prepare_train_data.py

数据增强: 可以跳过
PYTHONPATH=./ python data_process/data_augmentation.py

模型训练:
nohup sh run.sh&

推理:
PYTHONPATH=./ python extends/predictor.py

```

## 3. 对比学习

```
数据准备:
PYTHONPATH=./ python data_process/prepare_simcse_data.py

模型训练:
nohup python run_simcse.py 0&
```