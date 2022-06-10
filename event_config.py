import os
import sys


type_suffix = "event_1115" # 用于区分保存的模型
type_version = "1636965450" # 用于做预测的模型号
type_config = {
    "data_dir": "data/type_data", # 数据目录
    "model_type": "mrc", # 模型类型，“mrc” or "bert" or "casee"
    "label_nums": 25, # 类别数
    "max_seq_length": 512,
    
    "type_file": "event_types.txt", # 类别名称文件，其中包含要识别的所有类别名
    
    "train_data": "has_event_annotated_1115_proc.json", 
    "dev_data": "has_event_annotated_1110_proc.json",
    "test_data": "has_event_annotated_1110_proc.json",
    
    "vocab_file": "vocab.txt", 
    "bert_pretrained_model_path":os.path.join("bert","chinese_roberta_wwm_ext_L-12_H-768_A-12"),
    "bert_config_path": "bert_config.json",
    "bert_init_checkpoints": "bert_model.ckpt",

    "save_predict": os.path.join("data", "predict", type_suffix), # 保存预测结果的路径
    "model_dir": os.path.join("output", type_suffix, "checkpoint"),
    "model_pb": os.path.join("output", type_suffix, "saved_model"),
    # "best_model_dir": os.path.join("output", type_suffix, "checkpoint/export/best_exporter/{}/variables/variables".format(type_version)),  # 用来做预测的模型保存路径，做预测前根据output/model中保存的模型路径名修改该参数，这里选择的是best_exporter导出的模型
    "best_model_dir": os.path.join("output", type_suffix, "saved_model/{}/variables/variables".format(type_version)), # 采用最近保存的模型做预测，有时比保存下来最佳的模型表现好
    "best_model_pb": os.path.join("output", type_suffix, "saved_model/{}".format(type_version))
}


stat_suffix = "status_1201"
stat_version = "1638353080"
status_config = {
    "data_dir": "our_data/Event_1125_multi_task_1201/Status_data",
    "model_type": "bert",
    "label_nums": 4,
    "max_seq_length": 512,
    "task_pred_score":0.5,
    
    "bert_pretrained_model_path":os.path.join("bert","chinese_roberta_wwm_ext_L-12_H-768_A-12"),
    "bert_config_path": "bert_config.json",
    "bert_init_checkpoints": "bert_model.ckpt",
    "vocab_file": "vocab.txt",
    
    "type_file": "label_idx_map.txt",
    "train_data": "train.json",
    "dev_data": "dev.json",
    "test_data": "test.json",
    
    "save_predict": os.path.join("output", stat_suffix,"predict", "predict_result_{}".format(stat_suffix)),
    "model_dir": os.path.join("output", stat_suffix, "checkpoint"),
    "model_pb": os.path.join("output", stat_suffix, "saved_model"),
    #"best_model_dir": os.path.join("output", stat_suffix, "checkpoint/export/best_exporter", stat_version, "variables", "variables"),
    #"best_model_pb": os.path.join("output", stat_suffix,  "checkpoint/export/best_exporter/{}".format(stat_version))
    "best_model_dir": os.path.join("output", stat_suffix, "saved_model/{}/variables/variables".format(stat_version)), # 采用最近保存的模型做预测，有时比保存下来最佳的模型表现好
    "best_model_pb": os.path.join("output", stat_suffix, "saved_model/{}".format(stat_version))
}


role_suffix = "1117_role"
role_version = "1637226026"
role_config = {
    "data_dir": "data/role_data",
    "max_seq_length": 512,
    
    "train_data": "has_event_annotated_1115.json",
    "dev_data": "has_event_annotated_1110.json",
    "test_data": "has_event_annotated_1110.json",

    "bert_pretrained_model_path":os.path.join("bert","chinese_roberta_wwm_ext_L-12_H-768_A-12"),
    "bert_config_path": "bert_config.json",
    "bert_init_checkpoints": "bert_model.ckpt",
    "vocab_file": "vocab.txt",
    
    "event_schema": "event_schema_mt.json",
    "model_dir": os.path.join("output", role_suffix, "checkpoint"),
    "model_pb": os.path.join("output", role_suffix, "saved_model"),
    "best_model_dir": os.path.join("output", role_suffix, "checkpoint/export/best_exporter", role_version, "variables", "variables"),
    # "best_model_dir": os.path.join("output", role_suffix, "saved_model/{}/variables/variables".format(role_version)),
    "best_model_pb": os.path.join("output", role_suffix, "checkpoint/export/best_exporter/{}".format(role_version))
}

stat_suffix = "subject_1201"
stat_version = "1638350550"
subject_config = {
    "data_dir": "our_data/Event_1125_multi_task_1201/Subject_data",
    "model_type": "bert",
    "label_nums": 6,
    "max_seq_length": 512,
    "task_pred_score":0.999,
    
    "bert_pretrained_model_path":os.path.join("bert","chinese_roberta_wwm_ext_L-12_H-768_A-12"),
    "bert_config_path": "bert_config.json",
    "bert_init_checkpoints": "bert_model.ckpt",
    "vocab_file": "vocab.txt",
    
    "type_file": "label_idx_map.txt",
    "train_data": "train.json",
    "dev_data": "dev.json",
    "test_data": "dev.json",
    
    "save_predict": os.path.join("output", stat_suffix,"predict", "predict_result_{}".format(stat_suffix)),
    "model_dir": os.path.join("output", stat_suffix, "checkpoint"),
    "model_pb": os.path.join("output", stat_suffix, "saved_model"),
    #"best_model_dir": os.path.join("output", stat_suffix, "checkpoint/export/best_exporter", stat_version, "variables", "variables"),
    #"best_model_pb": os.path.join("output", stat_suffix,  "checkpoint/export/best_exporter/{}".format(stat_version))
    "best_model_dir": os.path.join("output", stat_suffix, "saved_model/{}/variables/variables".format(stat_version)), # 采用最近保存的模型做预测，有时比保存下来最佳的模型表现好
    "best_model_pb": os.path.join("output", stat_suffix, "saved_model/{}".format(stat_version))
}

stat_suffix = "object_1201"
stat_version = "1638351285"
object_config = {
    "data_dir": "our_data/Event_1125_multi_task_1201/Object_data",
    "model_type": "bert",
    "label_nums": 4,
    "max_seq_length": 512,
    "task_pred_score":0.999,
    
    "bert_pretrained_model_path":os.path.join("bert","chinese_roberta_wwm_ext_L-12_H-768_A-12"),
    "bert_config_path": "bert_config.json",
    "bert_init_checkpoints": "bert_model.ckpt",
    "vocab_file": "vocab.txt",
    
    "type_file": "label_idx_map.txt",
    "train_data": "train.json",
    "dev_data": "dev.json",
    "test_data": "test.json",
    
    "save_predict": os.path.join("output", stat_suffix,"predict", "predict_result_{}".format(stat_suffix)),
    "model_dir": os.path.join("output", stat_suffix, "checkpoint"),
    "model_pb": os.path.join("output", stat_suffix, "saved_model"),
    #"best_model_dir": os.path.join("output", stat_suffix, "checkpoint/export/best_exporter", stat_version, "variables", "variables"),
    #"best_model_pb": os.path.join("output", stat_suffix,  "checkpoint/export/best_exporter/{}".format(stat_version))
    "best_model_dir": os.path.join("output", stat_suffix, "saved_model/{}/variables/variables".format(stat_version)), # 采用最近保存的模型做预测，有时比保存下来最佳的模型表现好
    "best_model_pb": os.path.join("output", stat_suffix, "saved_model/{}".format(stat_version))
}

stat_suffix = "Event_1125_multi_task_1201"
stat_version = "1638344680"
multi_task_config = {
    "data_dir": "our_data/Event_1125_multi_task_1201",
    "model_type": "bert",
    "label_nums": -1,
    "max_seq_length": 512,
    "task_pred_score":[0.5,0.999,0.998],#顺序位置与type_file中label_type对应的label_tid一一对应
    "task_loss_weight":[0.5,1,1],#顺序位置与type_file中label_type对应的label_tid一一对应
    
    "bert_pretrained_model_path":os.path.join("bert","chinese_roberta_wwm_ext_L-12_H-768_A-12"),
    "bert_config_path": "bert_config.json",
    "bert_init_checkpoints": "bert_model.ckpt",
    "vocab_file": "vocab.txt",
    
    "type_file": "label_idx_map.txt",
    "train_data": "train.json",
    "dev_data": "dev.json",
    "test_data": "test.json",
    
    "save_predict": os.path.join("output", stat_suffix,"predict", "predict_result_{}".format(stat_suffix)),
    "model_dir": os.path.join("output", stat_suffix, "checkpoint"),
    "model_pb": os.path.join("output", stat_suffix, "saved_model"),
    #"best_model_dir": os.path.join("output", stat_suffix, "checkpoint/export/best_exporter", stat_version, "variables", "variables"),
    #"best_model_pb": os.path.join("output", stat_suffix,  "checkpoint/export/best_exporter/{}".format(stat_version))
    "best_model_dir": os.path.join("output", stat_suffix, "saved_model/{}/variables/variables".format(stat_version)), # 采用最近保存的模型做预测，有时比保存下来最佳的模型表现好
    "best_model_pb": os.path.join("output", stat_suffix, "saved_model/{}".format(stat_version))
}
#status_config=object_config
