import tensorflow as tf
import numpy as np
import common_utils
import sys
import math
from bert import modeling, optimization
logger = common_utils.set_logger('Training...')        

class BertAttriMultiTaskCLS(object):
    def __init__(self, params, bert_config):
        self.dropout_rate = params["dropout_prob"]
        self.num_class_list=params["num_class_list"]
        self.class_weight = params["class_weight"]
        self.task_loss_weight=params["task_loss_weight"]
        self.bert_config = bert_config
    

    def __call__(self, input_ids, labels, text_length_list, token_type_ids, is_training, is_testing=False):
        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            use_one_hot_embeddings=False, token_type_ids=token_type_ids
        )
        bert_output = bert_model.get_pooled_output()
        
        loss_all=None
        per_example_loss_list=[]
        pred_prob_list=[]
        probabilities_list=[]
        for label_type_id, num_class in enumerate(self.num_class_list): 
            bert_output = tf.layers.dropout(bert_output,rate=self.dropout_rate,training=is_training)
            bert_project = tf.layers.dense(bert_output, num_class)
            pred_prob = tf.nn.log_softmax(bert_project, axis=-1)
            probabilities = tf.nn.softmax(bert_project, axis=-1)
            pred_prob_list.append(pred_prob)
            probabilities_list.append(probabilities)
            if not is_testing:
                label=labels[label_type_id]
                # per_example_loss = - tf.reduce_sum(labels * pred_prob * self.class_weight, axis=-1)
                per_example_loss = -tf.reduce_sum(label * pred_prob, axis=-1)#只计算有标签位置上的loss
                loss = tf.reduce_mean(per_example_loss)
                per_example_loss_list.append(per_example_loss)
                if loss_all is None:
                    loss_all=loss*self.task_loss_weight[label_type_id]
                else:
                    loss_all+=(loss*self.task_loss_weight[label_type_id])
        if not is_testing:
            return per_example_loss_list, loss_all, pred_prob_list, probabilities_list
        else:
            return pred_prob_list, probabilities_list
        
def bert_multi_task_cls_model_fn_builder(bert_config_file, init_checkpoints, args):
    '''
    多任务学习。输入同一个句子，使用[CLS]位，连接多个linear层，进行多个分类任务[task1, task2, task3,...]，将多个loss相加
    注意点：
    1. 有的样本在task1上是没有标签的，那么需要mask掉这个样本在task1上的loss。因此需要增加输入：label mask
    2. 每个任务上样本不均衡，需要各自分配标签权重，
    '''
    
    
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        if isinstance(features, dict):
            features = features['words'], features['token_type_ids'], features['text_length']
        input_ids, token_type_ids, text_length_list = features
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_testing = (mode == tf.estimator.ModeKeys.PREDICT)
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        tag_model = BertAttriMultiTaskCLS(params, bert_config)
        if is_testing:
            pred_ids_list, probabilities_list = tag_model(input_ids, labels, text_length_list, token_type_ids,is_training, is_testing)
        else:
            per_example_loss_list, loss_all, pred_ids_list, probabilities_list = tag_model(input_ids, labels, text_length_list, token_type_ids, is_training)  

        tvars = tf.trainable_variables()
        
        # 加载BERT模型
        if init_checkpoints:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoints)
            tf.train.init_from_checkpoint(init_checkpoints, assignment_map)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            for i in tvars:
                print(i)
            print("*" * 15 + "Training" + "*" * 15)
#             train_op = optimization.create_optimizer(loss, args.lr, params["decay_steps"], None, False)
            train_op = optimization.create_optimizer(loss_all, args.lr, params["train_steps"], params["warmup_steps"], False)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss_all,
                train_op=train_op
            )

        elif mode == tf.estimator.ModeKeys.EVAL:
            print("*" * 15 + "Evaluation" + "*" * 15)
            pred_ids_list = [tf.argmax(pred_ids, axis=-1) for pred_ids in pred_ids_list]
            pred_ids_list = [tf.one_hot(pred_ids, depth=num_class, dtype=tf.float32) for num_class,pred_ids in zip(params["num_class_list"],pred_ids_list)]
            def metric_fn(per_example_loss_list, labels_list, probabilities_list):
                eval_dict = {}
                for label_id,(num_class, per_example_loss, probabilities, label_ids) in enumerate(zip(params["num_class_list"], per_example_loss_list, probabilities_list, labels_list)):
                    logits_split = tf.split(probabilities, num_class, axis=-1)
                    label_ids_split = tf.split(label_ids, num_class, axis=-1)
                    for j, logits in enumerate(logits_split):
                        label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                        precision, update_op_pre = tf.metrics.precision(label_id_, logits)
                        recall, update_op_recall = tf.metrics.recall(label_id_, logits)
                        f1 = (2 * precision * recall) / (precision + recall)
                        eval_dict[str(label_id)+'@'+str(j)+'-pre'] = (precision, update_op_pre)
                        eval_dict[str(label_id)+'@'+str(j)+'-rec'] = (recall, update_op_recall)
                        eval_dict[str(label_id)+'@'+str(j)+'-f1'] = (f1, tf.identity(f1))
                        
                    eval_dict[str(label_id)+'@'+'eval_loss'] = tf.metrics.mean(values=per_example_loss)
                    eval_dict[str(label_id)+'@'+'label_accuracy'] = tf.metrics.accuracy(labels=tf.argmax(label_ids, axis=-1), 
                                                                  predictions=tf.argmax(probabilities, axis=-1), name='acc_op')
                    mic_pre, mic_rec = tf.metrics.precision(label_ids, probabilities), tf.metrics.recall(label_ids, probabilities)
                    eval_dict[str(label_id)+'@'+'mic_pre']=tf.metrics.precision(label_ids, probabilities)
                    eval_dict[str(label_id)+'@'+'mic_rec']=tf.metrics.recall(label_ids, probabilities)
                    
                return eval_dict
            
            eval_metrics = metric_fn(per_example_loss_list, labels, pred_ids_list)
            output_spec = tf.estimator.EstimatorSpec(
                eval_metric_ops=eval_metrics,
                mode=mode,
                loss=loss_all
            )
        else:
            print('#@'*10)
            probabilities_list=tf.concat(probabilities_list, axis=-1)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=probabilities_list
            )
        return output_spec

    return model_fn