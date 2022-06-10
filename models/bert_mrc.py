import tensorflow as tf
import common_utils
from models.tf_metrics import precision, recall, f1
from bert import modeling, optimization
from models.utils import ce_loss,cal_binary_dsc_loss,dl_dsc_loss,vanilla_dsc_loss,focal_loss

logger = common_utils.set_logger('NER Training...')

class bertMRC(object):
    def __init__(self,params,bert_config):
        # 丢弃概率
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = 2
        self.bert_config = bert_config

    def __call__(self,input_ids,start_labels,end_labels,token_type_ids_list,query_len_list,text_length_list,is_training,is_testing=False):
        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            token_type_ids=token_type_ids_list,
            use_one_hot_embeddings=False
        )
        bert_seq_output = bert_model.get_sequence_output()

        # bert_project = tf.layers.dense(bert_seq_output, self.hidden_units, activation=tf.nn.relu)
        # bert_project = tf.layers.dropout(bert_project, rate=self.dropout_rate, training=is_training)
        start_logits = tf.layers.dense(bert_seq_output,self.num_labels)
        end_logits = tf.layers.dense(bert_seq_output, self.num_labels)
        query_span_mask = tf.cast(tf.sequence_mask(query_len_list),tf.int32)
        total_seq_mask = tf.cast(tf.sequence_mask(text_length_list),tf.int32)
        query_span_mask = query_span_mask * -1
        query_len_max = tf.shape(query_span_mask)[1]
        left_query_len_max = tf.shape(total_seq_mask)[1] - query_len_max
        zero_mask_left_span = tf.zeros((tf.shape(query_span_mask)[0],left_query_len_max),dtype=tf.int32)
        final_mask = tf.concat((query_span_mask,zero_mask_left_span),axis=-1)
        final_mask = final_mask + total_seq_mask
        predict_start_ids = tf.argmax(start_logits, axis=-1, name="pred_start_ids")
        predict_end_ids = tf.argmax(end_logits, axis=-1, name="pred_end_ids")
        if not is_testing:
            # one_hot_labels = tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)
            # start_loss = ce_loss(start_logits,start_labels,final_mask,self.num_labels,True)
            # end_loss = ce_loss(end_logits,end_labels,final_mask,self.num_labels,True)

            # focal loss
            start_loss = focal_loss(start_logits,start_labels,final_mask,self.num_labels,True)
            end_loss = focal_loss(end_logits,end_labels,final_mask,self.num_labels,True)

            final_loss = start_loss + end_loss
            return final_loss,predict_start_ids,predict_end_ids,final_mask
        else:
            return predict_start_ids,predict_end_ids,final_mask


def bert_mrc_model_fn_builder(bert_config_file,init_checkpoints,args):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        if isinstance(features, dict):
            features = features['words'],features['text_length'],features['query_length'],features['token_type_ids']
        print(features)
        input_ids,text_length_list,query_length_list,token_type_id_list = features
        if labels is not None:
            start_labels,end_labels = labels
        else:
            start_labels, end_labels = None,None
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_testing = (mode == tf.estimator.ModeKeys.PREDICT)
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        tag_model = bertMRC(params,bert_config)
        # input_ids,labels,token_type_ids_list,query_len_list,text_length_list,is_training,is_testing=False
        if is_testing:
            pred_start_ids,pred_end_ids,weight = tag_model(input_ids,start_labels,end_labels, token_type_id_list,
                                                           query_length_list, text_length_list, is_training,is_testing)
        else:
            loss,pred_start_ids,pred_end_ids,weight = tag_model(input_ids,start_labels,end_labels,token_type_id_list,
                                                                query_length_list,text_length_list,is_training)

        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoints:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoints)
            tf.train.init_from_checkpoint(init_checkpoints, assignment_map)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
#             train_op = optimization.create_optimizer(loss,args.lr, params["decay_steps"],args.clip_norm)
            train_op = optimization.create_optimizer(loss, args.lr, params["train_steps"], params["warmup_steps"], False)
#             hook_dict = {}
#             hook_dict['loss'] = loss
#             hook_dict['global_steps'] = tf.train.get_or_create_global_step()
#             logging_hook = tf.train.LoggingTensorHook(
#                 hook_dict, every_n_iter=args.print_log_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
#                 training_hooks=[logging_hook]
            )

        elif mode == tf.estimator.ModeKeys.EVAL:
            f1_start_val,f1_update_op_val = f1(labels=start_labels,predictions=pred_start_ids,num_classes=2,weights=weight,average="micro")
            f1_end_val,f1_end_update_op_val = f1(labels=end_labels,predictions=pred_end_ids,num_classes=2,weights=weight,average="micro")
            eval_metric_ops = {
            "f1_start_micro":(f1_start_val,f1_update_op_val),
            "f1_end_micro":(f1_end_val,f1_end_update_op_val)}
            output_spec = tf.estimator.EstimatorSpec(
                eval_metric_ops=eval_metric_ops,
                mode=mode,
                loss=loss
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"start_ids":pred_start_ids,"end_ids":pred_end_ids}
            )
        return output_spec
    return model_fn
