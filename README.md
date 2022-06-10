## 简介
事件抽取Pipeline模型，包含三个模块，事件类型识别（多标签分类），事件状态及主客体识别（多分类），事件论元抽取（抽取模型）

其中类型识别和主客体、状态的识别都是基于BERT的分类模型（多标签 or 多分类），以多标签分类为例，介绍一下目前三种模型方法的大致思路：
1. BERT多标签分类（model_type: bert）：将句子经BERT编码后的【CLS】向量拿出来做多次二分类
2. MRC多标签分类（model_type: mrc）：在句子后拼接所有类别名称，类别名称两侧用【unused1】和【unused2】分隔，经BERT编码后将每个类别编码后的分隔符【unused1】拿出分别做二分类
3. CasEE多标签分类（model_type: casee）：初始化一个包含所有类别向量的矩阵，每个类别对应不同的向量c，通过不同的类别向量c来与经BERT编码后的句子做attention，以此来捕捉与该类别最相关的词汇，并对attention后的文本表示来做分类

论元抽取目前采用的是基于BERT的MRC抽取范式，对于每种事件的角色，生成相应的query，如“找出 <下单> 事件的 <下单商家>”，拼接在文本上并进行片段的抽取


## 数据格式

1. 标注数据格式（也是事件类型识别和论元抽取的训练数据格式）
```
{'text': '嗯，就说是您这边购买的那个尊宝披萨的一九第一湾店这个订单对吗女士？',
 'event_list': [{'event_type': '下单',
                 'status': '完成',
                 'arguments': [{'argument': '购买', 'argument_start_index': 9, 'role': '触发词'},
                               {'argument': '尊宝披萨的一九第一湾店', 'argument_start_index': 14, 'role': '下单商家'}]
                }]
}
```

2. 事件状态识别训练数据 （主客体也可参考这种方式）
```
{'text': '退单事件：嗯，那您看是这样的，如果您表示说是系统的一个问题，现在骑手确实没有办法再次给您一个配送，那您看这样吧，这个订单的话现在我立即帮您向咱们后台的话进行个反馈，我们去核实下是否是咱们这边定位那个问题，那您看东西确实现在可能没有办法及时再次给您配送到了，然后我在后台给您取消一个订单，您这边辛苦您再重新下一份儿餐品，您看行吗？',
 'event_list': [{'event_type': '未开始'}]}
```
注：实际这里的event指的是status，是为了数据处理上的统一

3. 多任务分类训练数据

目前只支持多分类，不支持多标签分类，因此输入的标签每个type只能有一个值
3.1 训练集：
```
{'text':'xxxxx','event_list': [{'type_1': 'aaa'},{'type_2':'bbb'},{'type_3':'ccc'}]}

```
3.2 标签索引文件：
```
骑手	0	type_1	0
商家	1	type_1	0
订单	0	type_2	1
系统	1	type_2	1
完成	0	type_3	2
未完成	1	type_3	2
进行中	2	type_3	2
```
每行表示一个标签，第三列表示该标签是哪种标签label_type（即，type_1~type_3中的哪一个），第四列表示该标签种类label_type的索引label_tid, 第一列表示标签名，第二列表示该标签在该标签种类下的index



## 主要文件介绍

1. 配置文件：event_config.py
   - 其中包括**模型方法**、**类别数目**、**训练数据路径**、**模型存放路径**、**预测模型路径**等参数的设置
2. 模型文件 (models)
    - 多标签分类：bert_type_classification.py
    - 多分类：multi_class_model.py
    - 基于BERT的MRC抽取模型：bert_mrc.py
3. 数据处理 (data_processing)：event_data_prepare.py
4. 训练框架文件：train_helper.py

## 模型训练和预测

以事件类型识别为例，其余在训练命令中只需修改相应的module_name为status或role。

### 模型训练
```bash
CUDA_VISIBLE_DEVICES=2 nohup python run_event.py --model_type type --dropout_prob 0.1 --epochs 10 --lr 2e-5 --train_batch_size 16 --valid_batch_size 4 --do_train >type_cls.log 2>&1 &
```

### 模型预测（暂不支持事件论元抽取）

根据保存模型的存放位置，修改event_config.py中用于预测的best_model_dir及version后运行以下命令
```bash
CUDA_VISIBLE_DEVICES=2 python run_event.py --model_type type --do_test
```

# 依赖

General

- Python (verified on 3.6)
- CUDA (verified on 10.0)

Python Packages

- see requirements.txt



# 参考

https://github.com/qiufengyuyi/event_extraction
https://github.com/JiaweiSheng/CasEE
