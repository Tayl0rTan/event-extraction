import os
import numpy as np
import time
import logging
import json

from common_utils import set_logger
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.bert_mrc import bert_mrc_model_fn_builder
from models.multi_class_model import bert_multicls_model_fn_builder
from models.bert_event_type_classification import bert_classification_model_fn_builder
from models.multi_task import bert_multi_task_cls_model_fn_builder
from data_processing.data_utils import *
from data_processing.event_data_prepare import ClassifyDataPrepare, EventRolePrepareMRC, MultiTaskClassifyDataPrepare, event_input_bert_mrc_fn, event_index_class_input_bert_fn, event_multi_task_input_bert_fn
from event_config import type_config, status_config, role_config, multi_task_config

logger = set_logger("[run training]")

def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    words_seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words_seq')
    receiver_tensors = {'words': words, 'text_length': nwords, 'words_seq': words_seq}
    features = {'words': words, 'text_length': nwords, 'words_seq': words_seq}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def bert_serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    receiver_tensors = {'words': words, 'text_length': nwords}
    features = {'words': words, 'text_length': nwords}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def bert_type_serving_input_receiver_fn():
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    token_type_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="token_type_ids")
    type_index_ids = tf.placeholder(dtype=tf.int32, shape=[None, type_config.get("label_nums")], name="type_index_in_ids_list")
    receiver_tensors = {'words': words, 'text_length': nwords, 'token_type_ids': token_type_ids,
                        'type_index_in_ids_list': type_index_ids}
    features = {'words': words, 'text_length': nwords, 'token_type_ids': token_type_ids,
                'type_index_in_ids_list': type_index_ids}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def bert_status_serving_input_receiver_fn():
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    token_type_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="token_type_ids")
    type_index_ids = tf.placeholder(dtype=tf.int32, shape=[None, status_config.get("label_nums")], name="type_index_in_ids_list")
    receiver_tensors = {'words': words, 'text_length': nwords, 'token_type_ids': token_type_ids,
                        'type_index_in_ids_list': type_index_ids}
    features = {'words': words, 'text_length': nwords, 'token_type_ids': token_type_ids,
                'type_index_in_ids_list': type_index_ids}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def bert_mrc_serving_input_receiver_fn():
    # features['words'],features['text_length'],features['query_length'],features['token_type_ids']
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    query_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="query_length")
    token_type_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="token_type_ids")
    receiver_tensors = {'words': words, 'text_length': nwords, 'query_length': query_lengths,
                        'token_type_ids': token_type_ids}
    features = {'words': words, 'text_length': nwords, 'query_length': query_lengths, 'token_type_ids': token_type_ids}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def get_mrc_input_example(data_dir, set_name='train'):
    data_list = np.load("{}/token_ids_{}.npy".format(data_dir, set_name), allow_pickle=True)
    label_start_list = np.load("{}/labels_start_{}.npy".format(data_dir, set_name), allow_pickle=True)
    label_end_list = np.load("{}/labels_end_{}.npy".format(data_dir, set_name), allow_pickle=True)
    token_type_id_list = np.load("{}/token_type_ids_{}.npy".format(data_dir, set_name), allow_pickle=True)
    query_len_list = np.load("{}/query_lens_{}.npy".format(data_dir, set_name), allow_pickle=True)
    return data_list, label_start_list, label_end_list, token_type_id_list, query_len_list

def get_input_example(data_dir, set_name='train'):
    data_list = np.load("{}/token_ids_{}.npy".format(data_dir, set_name), allow_pickle=True)
    label_list = np.load("{}/labels_{}.npy".format(data_dir, set_name), allow_pickle=True)
    token_type_id_list = np.load("{}/token_type_ids_{}.npy".format(data_dir, set_name), allow_pickle=True)
    type_index_ids_list = np.load("{}/type_index_in_token_ids_{}.npy".format(data_dir, set_name), allow_pickle=True)
    return data_list, label_list, token_type_id_list, type_index_ids_list

def get_multi_task_input_example(data_dir, label_tid2type_map, set_name='train'):
    data_list = np.load("{}/token_ids_{}.npy".format(data_dir, set_name), allow_pickle=True)
    
    label_list=[None for _ in range(max(label_tid2type_map.keys())+1)]
    for idx in label_tid2type_map:
        label_type=label_tid2type_map[idx]
        label_list[idx]=np.load("{}/labels_{}_{}.npy".format(data_dir,label_type, set_name), allow_pickle=True)
    token_type_id_list = np.load("{}/token_type_ids_{}.npy".format(data_dir, set_name), allow_pickle=True)
    return data_list, label_list, token_type_id_list

def run_event_classification(args):
    """
    事件类型识别，多标签二分类问题
    :param args:
    :return:
    """
    model_base_dir = type_config.get('model_dir')
    pb_model_dir = type_config.get('model_pb')
    data_dir = type_config.get("data_dir")

    data_loader = ClassifyDataPrepare(type_config)
    
    train_data_list, train_label_list, train_token_type_id_list, train_type_index_ids_list = get_input_example(data_dir, 'train')
    dev_data_list, dev_label_list, dev_token_type_id_list, dev_type_index_ids_list = get_input_example(data_dir, 'dev')
    test_data_list, test_label_list, test_token_type_id_list, test_type_index_ids_list = get_input_example(data_dir, 'test')

    # 根据各个类别样本数得到class_weight，用于缓解样本不均衡的问题
    train_labels = np.array(train_label_list)
    a = np.sum(train_labels, axis=0)
    weights = []
    max_num = max(a)
    for ele in a:
        prop = min(max_num / ele, 10) if ele != 0 else 0
        weights.append(prop)
    class_weight = np.array(weights)
    class_weight = np.reshape(class_weight, (1, type_config.get("label_nums")))

    
    train_samples_nums = len(train_data_list)
    dev_samples_nums = len(dev_data_list)
    if train_samples_nums % args.train_batch_size != 0:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size)
    logger.info('*****train_set sample nums:{}'.format(train_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps * args.epochs
    warmup_steps = int(train_steps_nums * 0.1)
    logger.info('*****warm up steps:{}'.format(warmup_steps))
    params = {"dropout_prob": args.dropout_prob, "num_labels": data_loader.labels_map_len, "model_type": type_config.get("model_type"),
              "warmup_steps": warmup_steps, "class_weight": class_weight, "train_steps": train_steps_nums}
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=model_base_dir,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=3,
    )
    bert_init_checkpoints = os.path.join(type_config.get("bert_pretrained_model_path"),
                                         type_config.get("bert_init_checkpoints"))
    bert_config_path = os.path.join(type_config.get("bert_pretrained_model_path"),
                                    type_config.get("bert_config_path"))
    model_fn = bert_classification_model_fn_builder(bert_config_path, bert_init_checkpoints, args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    if args.do_train:

        train_input_fn = lambda: event_index_class_input_bert_fn(train_data_list,
                                                                 token_type_ids=train_token_type_id_list,
                                                                 type_index_ids_list=train_type_index_ids_list,
                                                                 label_map_len=data_loader.labels_map_len,
                                                                 is_training=True, is_testing=False, args=args,
                                                                 input_Ys=train_label_list)

        eval_input_fn = lambda: event_index_class_input_bert_fn(dev_data_list, token_type_ids=dev_token_type_id_list,
                                                                type_index_ids_list=dev_type_index_ids_list,
                                                                label_map_len=data_loader.labels_map_len,
                                                                is_training=False, is_testing=False, args=args,
                                                                input_Ys=dev_label_list)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums
                                            )
        exporter = tf.estimator.BestExporter(exports_to_keep=1,
                                             serving_input_receiver_fn=bert_type_serving_input_receiver_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=0, exporters=[exporter])
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        estimator.export_saved_model(pb_model_dir, bert_type_serving_input_receiver_fn)
        
    if args.do_test:
        test_input_fn = lambda: event_index_class_input_bert_fn(test_data_list, token_type_ids=test_token_type_id_list,
                                                                type_index_ids_list=test_type_index_ids_list,
                                                                label_map_len=data_loader.labels_map_len,
                                                                is_training=False, is_testing=True, args=args,
                                                                input_Ys=test_label_list)
        predictions = estimator.predict(input_fn=test_input_fn, checkpoint_path=type_config.get('best_model_dir'))
        
        def get_prf_for_each_class(preds, labels):
            """ 计算各个类别及汇总的prf
            """
            labels_map = data_loader.id2labels_map
            pre_list = precision_score(labels, preds, average=None)
            rec_list = recall_score(labels, preds, average=None)
            f1_list = f1_score(labels, preds, average=None)
            mic_pre, mic_rec, mic_f1 = precision_score(labels, preds, average="micro"), recall_score(labels, preds, average="micro"), f1_score(labels, preds, average="micro")
            metric_dic = {}
            for i in range(type_config.get("label_nums")):
                label, pre, rec, f1 = labels_map[i], pre_list[i], rec_list[i], f1_list[i]
                metric_dic[label] = {'pre': round(pre*100, 2), 'rec': round(rec*100, 2), 'f1': round(f1*100, 2)}
            metric_dic['Micro'] = {'mic_pre': round(mic_pre*100, 2), 'mic_rec': round(mic_rec*100, 2), 'mic_f1': round(mic_f1*100, 2)}
            metric_dic['sent_accuracy'] = round(accuracy_score(labels, preds) * 100, 2)
            return metric_dic
        
        predictions = np.array(list(predictions))
        np.save(type_config.get('save_predict') + '.npy', predictions)
        preds = np.where(predictions > 0.5, np.ones_like(predictions), np.zeros_like(predictions))
        metric_dic = get_prf_for_each_class(preds, test_label_list)
        print(json.dumps(metric_dic, ensure_ascii=False, indent=2))
        json.dump(metric_dic, open(type_config.get('save_predict') + '.json', 'w'))

def run_multi_classification(args):
    """
    多分类模型
    :param args:
    :return:
    """
    model_base_dir = status_config.get('model_dir')
    pb_model_dir = status_config.get('model_pb')
    data_dir = status_config.get("data_dir")
    
    data_loader = ClassifyDataPrepare(status_config)
    
    train_data_list, train_label_list, train_token_type_id_list, train_type_index_ids_list = get_input_example(data_dir, 'train')
    dev_data_list, dev_label_list, dev_token_type_id_list, dev_type_index_ids_list = get_input_example(data_dir, 'dev')
    test_data_list, test_label_list, test_token_type_id_list, test_type_index_ids_list = get_input_example(data_dir, 'test')

    # 根据各个类别样本数得到class_weight，用于缓解样本不均衡的问题
    train_labels = np.array(train_label_list)
    a = np.sum(train_labels, axis=0)
    weights = []
    max_num = max(a)
    for ele in a:
        prop = min(max_num / ele, 10) if ele != 0 else 0
        weights.append(prop)
    class_weight = np.array(weights)
    class_weight = np.reshape(class_weight, (1, status_config.get("label_nums")))

    print(class_weight)
    train_samples_nums = len(train_data_list)
    dev_samples_nums = len(dev_data_list)
    if train_samples_nums % args.train_batch_size != 0:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size)
    logger.info('*****train_set sample nums:{}'.format(train_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps * args.epochs
    warmup_steps = int(train_steps_nums * 0.1)
    logger.info('*****warm up steps:{}'.format(warmup_steps))
    params = {"dropout_prob": args.dropout_prob, "num_labels": data_loader.labels_map_len, "model_type": status_config.get("model_type"),
              "warmup_steps": warmup_steps, "class_weight": class_weight, "train_steps": train_steps_nums}
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=model_base_dir,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=3,
        # train_distribute=dist_strategy

    )
    bert_init_checkpoints = os.path.join(status_config.get("bert_pretrained_model_path"),
                                         status_config.get("bert_init_checkpoints"))
    bert_config_file = os.path.join(status_config.get("bert_pretrained_model_path"),
                                    status_config.get("bert_config_path"))
    
    model_fn = bert_multicls_model_fn_builder(bert_config_file, bert_init_checkpoints, args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    if args.do_train:
        train_input_fn = lambda: event_index_class_input_bert_fn(train_data_list,
                                                                 token_type_ids=train_token_type_id_list,
                                                                 type_index_ids_list=train_type_index_ids_list,
                                                                 label_map_len=data_loader.labels_map_len,
                                                                 is_training=True, is_testing=False, args=args,
                                                                 input_Ys=train_label_list)

        eval_input_fn = lambda: event_index_class_input_bert_fn(dev_data_list, token_type_ids=dev_token_type_id_list,
                                                                type_index_ids_list=dev_type_index_ids_list,
                                                                label_map_len=data_loader.labels_map_len,
                                                                is_training=False, is_testing=False, args=args,
                                                                input_Ys=dev_label_list)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums
                                            )
        exporter = tf.estimator.BestExporter(exports_to_keep=1,
                                             serving_input_receiver_fn=bert_status_serving_input_receiver_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=0, exporters=[exporter])
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        estimator.export_saved_model(pb_model_dir, bert_status_serving_input_receiver_fn)
        
    if args.do_test:
        test_input_fn = lambda: event_index_class_input_bert_fn(test_data_list, token_type_ids=test_token_type_id_list,
                                                                type_index_ids_list=test_type_index_ids_list,
                                                                label_map_len=data_loader.labels_map_len,
                                                                is_training=False, is_testing=True, args=args,
                                                                input_Ys=test_label_list)
        predictions = estimator.predict(input_fn=test_input_fn, checkpoint_path=status_config.get('best_model_dir'))

        def get_prf_for_each_class(preds, labels):
            labels_map = data_loader.id2labels_map
            pre_list = precision_score(labels, preds, average=None)
            rec_list = recall_score(labels, preds, average=None)
            f1_list = f1_score(labels, preds, average=None)
            mic_pre, mic_rec, mic_f1 = precision_score(labels, preds, average="micro"), recall_score(labels, preds, average="micro"), f1_score(labels, preds, average="micro")
            metric_dic = {}
            for i in range(labels.shape[-1]):
                label, pre, rec, f1 = labels_map[i], pre_list[i], rec_list[i], f1_list[i]
                metric_dic[label] = {'pre': round(pre*100, 2), 'rec': round(rec*100, 2), 'f1': round(f1*100, 2)}
            metric_dic['Micro'] = {'mic_pre': round(mic_pre*100, 2), 'mic_rec': round(mic_rec*100, 2), 'mic_f1': round(mic_f1*100, 2)}
            metric_dic['sent_accuracy'] = round(accuracy_score(labels, preds) * 100, 2)
            return metric_dic
        
        predictions = np.array(list(predictions))
        np.save(status_config.get('save_predict') + '.npy', predictions)
        pred_ids = np.argmax(predictions, axis=-1)

        pred_ids = np.eye(params["num_labels"])[pred_ids]
        pred_ids = np.where(predictions > status_config['task_pred_score'], pred_ids, np.zeros_like(pred_ids))

        #test_label_list = np.argmax(test_label_list, axis=-1)
        metric_dic = get_prf_for_each_class(pred_ids, test_label_list)
        
        print(json.dumps(metric_dic, ensure_ascii=False, indent=2))
        json.dump(metric_dic, open(status_config.get('save_predict') + '.json', 'w'))
        

def run_event_role_mrc(args):
    """
    baseline 用mrc来做事件role抽取
    :param args:
    :return:
    """
    model_base_dir = role_config.get('model_dir')
    pb_model_dir = role_config.get('model_pb')
    vocab_file_path = os.path.join(role_config.get("bert_pretrained_model_path"), role_config.get("vocab_file"))

    data_dir = role_config.get("data_dir")
    schema_file = os.path.join(data_dir, role_config.get("event_schema"))
    
    train_datas, train_start_labels, train_end_labels, train_token_type_id_list, train_query_lens = get_mrc_input_example(data_dir, 'train')
    dev_datas, dev_start_labels, dev_end_labels, dev_token_type_id_list, dev_query_lens = get_mrc_input_example(data_dir, 'dev')
    
    train_samples_nums = len(train_datas)
    dev_samples_nums = len(dev_datas)
    if train_samples_nums % args.train_batch_size != 0:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size)
    logger.info('*****train_set sample nums:{}'.format(train_samples_nums))
    logger.info('*****dev_set sample nums:{}'.format(dev_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps * args.epochs
    logger.info('*****train_total_steps:{}'.format(train_steps_nums))
    warmup_steps = int(train_steps_nums * 0.1)
    logger.info('*****warmup steps:{}'.format(warmup_steps))
    params = {"dropout_prob": args.dropout_prob, "train_steps": train_steps_nums, "warmup_steps": warmup_steps}
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=model_base_dir,
        save_summary_steps=each_epoch_steps,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=3,
    )
    bert_init_checkpoints = os.path.join(role_config.get("bert_pretrained_model_path"),
                                         role_config.get("bert_init_checkpoints"))
    bert_config_file = os.path.join(role_config.get("bert_pretrained_model_path"),
                                    role_config.get("bert_config_path"))
    model_fn = bert_mrc_model_fn_builder(bert_config_file, bert_init_checkpoints, args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)
    if args.do_train:
        train_input_fn = lambda: event_input_bert_mrc_fn(train_datas, train_start_labels, train_end_labels, 
                                                         train_token_type_id_list, train_query_lens,is_training=True, is_testing=False, args=args)
        eval_input_fn = lambda: event_input_bert_mrc_fn(train_datas, train_start_labels, train_end_labels, 
                                                        train_token_type_id_list, train_query_lens, is_training=False, is_testing=False, args=args)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums
                                            )
        exporter = tf.estimator.BestExporter(exports_to_keep=1,
                                             serving_input_receiver_fn=bert_mrc_serving_input_receiver_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, exporters=[exporter], throttle_secs=0)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        estimator.export_saved_model(pb_model_dir, bert_mrc_serving_input_receiver_fn)

        
def run_multi_task_classification(args):
    """
    多任务+多分类模型
    :param args:
    :return:
    """
    model_base_dir = multi_task_config.get('model_dir')
    pb_model_dir = multi_task_config.get('model_pb')
    data_dir = multi_task_config.get("data_dir")
    
    data_loader = MultiTaskClassifyDataPrepare(multi_task_config)
    label_tid2type_map=data_loader.label_tid2type_map
    print('标签类别对应id：',label_tid2type_map)
    
    train_data_list, train_label_list, train_token_type_id_list = get_multi_task_input_example(data_dir,label_tid2type_map, 'train')
    dev_data_list, dev_label_list, dev_token_type_id_list = get_multi_task_input_example(data_dir,label_tid2type_map, 'dev')
    test_data_list, test_label_list, test_token_type_id_list = get_multi_task_input_example(data_dir,label_tid2type_map, 'test')

    # 根据各个类别样本数得到class_weight，用于缓解样本不均衡的问题
    class_weight_list=[]
    for train_labels in train_label_list:
        train_labels = np.array(train_labels)
        a = np.sum(train_labels, axis=0)
        weights = []
        max_num = max(a)
        for ele in a:
            prop = min(max_num / ele, 10) if ele != 0 else 0
            weights.append(prop)
        class_weight = np.array(weights)
        class_weight = np.reshape(class_weight, (1, multi_task_config.get("label_nums")))
        print(class_weight)
        class_weight_list.append(class_weight)
        
    train_samples_nums = len(train_data_list)
    dev_samples_nums = len(dev_data_list)
    if train_samples_nums % args.train_batch_size != 0:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size)
    logger.info('*****train_set sample nums:{}'.format(train_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps * args.epochs
    warmup_steps = int(train_steps_nums * 0.1)
    logger.info('*****warm up steps:{}'.format(warmup_steps))
    params = {"dropout_prob": args.dropout_prob,"num_class_list":data_loader.label_tid2len_list, "model_type": multi_task_config.get("model_type"),
              "warmup_steps": warmup_steps, "class_weight": class_weight_list, "train_steps": train_steps_nums, "task_loss_weight":multi_task_config.get("task_loss_weight")}
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=model_base_dir,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=5,
        # train_distribute=dist_strategy

    )
    bert_init_checkpoints = os.path.join(multi_task_config.get("bert_pretrained_model_path"),
                                         multi_task_config.get("bert_init_checkpoints"))
    bert_config_file = os.path.join(multi_task_config.get("bert_pretrained_model_path"),
                                    multi_task_config.get("bert_config_path"))
    
    model_fn = bert_multi_task_cls_model_fn_builder(bert_config_file, bert_init_checkpoints, args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    if args.do_train:
        train_input_fn = lambda: event_multi_task_input_bert_fn(train_data_list,
                                                                 token_type_ids=train_token_type_id_list,
                                                                 is_training=True, is_testing=False, args=args,
                                                                 input_Ys=train_label_list, label_num=len(label_tid2type_map))

        eval_input_fn = lambda: event_multi_task_input_bert_fn(dev_data_list, 
                                                                token_type_ids=dev_token_type_id_list,
                                                                is_training=False, is_testing=False, args=args,
                                                                input_Ys=dev_label_list, label_num=len(label_tid2type_map))
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums
                                            )
        exporter = tf.estimator.BestExporter(exports_to_keep=1,
                                             serving_input_receiver_fn=bert_status_serving_input_receiver_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=0, exporters=[exporter])
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        estimator.export_saved_model(pb_model_dir, bert_status_serving_input_receiver_fn)
        
    if args.do_test:
        test_input_fn = lambda: event_multi_task_input_bert_fn(test_data_list, 
                                                                token_type_ids=test_token_type_id_list,
                                                                is_training=False, is_testing=True, args=args,
                                                                input_Ys=test_label_list, label_num=len(label_tid2type_map))
        predictions_list = estimator.predict(input_fn=test_input_fn, checkpoint_path=multi_task_config.get('best_model_dir'))
        #predictions_list = estimator.predict(input_fn=test_input_fn)

        def get_prf_for_each_class(preds, labels, labels_map):
            metric_dic = {}
            pre_list = precision_score(labels, preds, average=None)
            rec_list = recall_score(labels, preds, average=None)
            f1_list = f1_score(labels, preds, average=None)
            mic_pre, mic_rec, mic_f1 = precision_score(labels, preds, average="micro"), recall_score(labels, preds, average="micro"), f1_score(labels, preds, average="micro")
            
            for i in range(labels.shape[-1]):
                label, pre, rec, f1 = labels_map[i], pre_list[i], rec_list[i], f1_list[i]
                metric_dic[label] = {'pre': round(pre*100, 2), 'rec': round(rec*100, 2), 'f1': round(f1*100, 2)}
            metric_dic['Micro'] = {'mic_pre': round(mic_pre*100, 2), 'mic_rec': round(mic_rec*100, 2), 'mic_f1': round(mic_f1*100, 2)}
            metric_dic['sent_accuracy'] = round(accuracy_score(labels, preds) * 100, 2)
            return metric_dic
        predictions_np = np.array(list(predictions_list))
        class_split_idx=[sum(data_loader.label_tid2len_list[:tid+1]) for tid in range(len(data_loader.label_tid2len_list))]
        predictions_list=np.split(predictions_np, class_split_idx, axis=-1)[:-1]
        np.save(multi_task_config.get('save_predict') + '.npy', predictions_np)
        pred_ids_list = [np.argmax(predictions, axis=-1) for predictions in predictions_list]

        pred_ids_list = [np.eye(num_labels)[pred_ids] for pred_ids, num_labels in zip(pred_ids_list, data_loader.label_tid2len_list)]
        pred_ids_list = [np.where(predictions > threshold, pred_ids, np.zeros_like(pred_ids)) for predictions, pred_ids,threshold in zip(predictions_list, pred_ids_list, multi_task_config['task_pred_score'])]

        #test_label_list = [np.argmax(test_label, axis=-1) for test_label in test_label_list]
        for label_type_idx, (preds, labels) in enumerate(zip(pred_ids_list, test_label_list)):
            label_type=data_loader.label_tid2type_map[label_type_idx]
            labels_map=data_loader.id2labels_map[label_type]
            print('*'*10+label_type+'*'*10)
            print('labels:', labels_map)
            metric_dic = get_prf_for_each_class(preds, labels, labels_map)
            print(json.dumps(metric_dic, ensure_ascii=False, indent=2))
            json.dump(metric_dic, open(multi_task_config.get('save_predict') +'_{}.json'.format(label_type), 'w'))
