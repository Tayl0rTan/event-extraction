import tensorflow as tf
import numpy as np
import common_utils
import sys
import math
from bert import modeling, optimization
logger = common_utils.set_logger('Training...')
        
class BertAttriCLS(object):
    def __init__(self, params, bert_config):
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = params["num_labels"]
        self.class_weight = params["class_weight"]
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
        bert_output = tf.layers.dropout(bert_output,rate=self.dropout_rate,training=is_training)
        bert_project = tf.layers.dense(bert_output, self.num_labels)
        pred_prob = tf.nn.log_softmax(bert_project, axis=-1)
        probabilities = tf.nn.softmax(bert_project, axis=-1)
        if not is_testing:
            # per_example_loss = - tf.reduce_sum(labels * pred_prob * self.class_weight, axis=-1)
            per_example_loss = -tf.reduce_sum(labels * pred_prob, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return per_example_loss, loss, pred_prob, probabilities
        else:
            return pred_prob, probabilities

class BertAttriCLSModified(object):
    def __init__(self, params, bert_config):
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = params["num_labels"]
        self.class_weight = params["class_weight"]
        self.bert_config = bert_config

    def __call__(self, input_ids, labels, text_length_list, token_type_ids, type_index_in_token_ids, is_training,
                 is_testing=False):

        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            use_one_hot_embeddings=False, token_type_ids=token_type_ids
        )
        bert_embedding = bert_model.get_sequence_output()
        bert_embedding = tf.layers.dropout(bert_embedding,rate=self.dropout_rate,training=is_training)
        batch_ids = tf.range(0, tf.shape(bert_embedding)[0])
        batch_ids = tf.expand_dims(batch_ids, 1)
        batch_ids = tf.expand_dims(batch_ids, -1)
        batch_ids = tf.tile(batch_ids, [1, self.num_labels, 1])
        type_index_in_token_ids = tf.expand_dims(type_index_in_token_ids, axis=-1)
        type_index = tf.concat([batch_ids, type_index_in_token_ids], axis=-1)
        type_head_tensor = tf.gather_nd(bert_embedding, type_index)
        bert_project = tf.layers.dense(type_head_tensor, 1)
        bert_project = tf.squeeze(bert_project, axis=-1)

        pred_prob = tf.nn.log_softmax(bert_project, axis=-1)
        probabilities = tf.nn.softmax(bert_project, axis=-1)

        if not is_testing:
#             per_example_loss = -tf.reduce_sum(labels * pred_prob * self.class_weight, axis=-1)
            per_example_loss = -tf.reduce_sum(labels * pred_prob, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return per_example_loss, loss, pred_prob, probabilities
        else:
            return pred_prob, probabilities
        
class BertAttriCLSAdaptive(object):
    def __init__(self, params, bert_config):
        rng = np.random.RandomState(3435)
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = params["num_labels"]
        self.class_weight = params["class_weight"]
        self.bert_config = bert_config
        self.hidden_size = bert_config.hidden_size
        evt_emb = np.asarray(0.01 * rng.standard_normal(size=(self.num_labels ,self.hidden_size)),dtype=np.float32)
        self.type_emb = tf.Variable(evt_emb,name='type_emb')

    
    def __call__(self, input_ids, labels, text_length_list, token_type_ids, is_training, is_testing=False):

        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            use_one_hot_embeddings=False, token_type_ids=token_type_ids
        )
        context = bert_model.get_sequence_output()
        context = tf.expand_dims(context, 1)
        context = tf.tile(context, (1, self.num_labels, 1, 1))
        type_emb = tf.expand_dims(self.type_emb, 0)
        type_emb = tf.expand_dims(type_emb, 2)
        type_emb = tf.broadcast_to(type_emb, tf.shape(context))
        concat_hidden = tf.concat([type_emb, context, tf.abs(type_emb - context), type_emb * context], axis=-1)
        project_hidden = tf.layers.dense(concat_hidden, self.hidden_size * 4, name='dense1')
        project_hidden = tf.tanh(project_hidden)
        scores = tf.layers.dense(project_hidden, 1, name='dense2')
        scores = tf.layers.dropout(scores,rate=self.dropout_rate,training=is_training)
        mask = tf.sequence_mask(lengths=text_length_list, dtype=tf.int32)
        mask = tf.expand_dims(mask, 1)
        mask = tf.expand_dims(mask, 3)
        mask = tf.broadcast_to(mask, tf.shape(scores))
        mask = tf.cast(mask, tf.float32)
        
        def mask_fill_inf(matrix, mask):
            negmask = 1 - mask
            num = 3.4 * math.pow(10, 38)
            return (matrix * mask) + (-((negmask * num + num) - num))
        
        scores = mask_fill_inf(scores, mask)
        scores = tf.transpose(scores, [0, 1, 3, 2])
        scores = tf.nn.softmax(scores, axis=-1)
        g = tf.matmul(scores, context)
        g = tf.squeeze(g, 2)
        type_emb_2 = tf.expand_dims(self.type_emb, 0)
        type_emb_2 = tf.broadcast_to(type_emb_2, tf.shape(g))
        concat_hidden_2 = tf.concat([type_emb_2, g, tf.abs(type_emb_2 - g), type_emb_2 * g], axis=-1)
        project_hidden_2 = tf.layers.dense(concat_hidden_2, self.hidden_size * 4, name='dense1', reuse=True)
        project_hidden_2 = tf.tanh(project_hidden_2)
        context_type_awared = tf.layers.dense(project_hidden_2, 1, name='dense2', reuse=True)
        context_type_awared = tf.squeeze(context_type_awared, -1)
        pred_prob = tf.nn.log_softmax(context_type_awared, axis=-1)
        probabilities = tf.nn.softmax(context_type_awared, axis=-1)

        if not is_testing:
            per_example_loss = -tf.reduce_sum(labels * pred_prob * self.class_weight, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            return per_example_loss, loss, pred_prob, probabilities
        else:
            return pred_prob, probabilities
        
        
def bert_multicls_model_fn_builder(bert_config_file, init_checkpoints, args):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        if isinstance(features, dict):
            features = features['words'], features['token_type_ids'], features['text_length'], features[
                'type_index_in_ids_list']
        # input_ids,token_type_ids,text_length_list = features
        input_ids, token_type_ids, text_length_list, type_index_in_ids_list = features
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_testing = (mode == tf.estimator.ModeKeys.PREDICT)
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        model_type = params["model_type"]
        assert model_type in ["mrc", "casee", "bert"], "Undefined model type {}".format(model_type)
        if model_type == "mrc":
            tag_model = BertAttriCLSModified(params, bert_config)  
        elif model_type == "casee": 
            tag_model = BertAttriCLSAdaptive(params,bert_config)
        else:
            tag_model = BertAttriCLS(params, bert_config)
        if is_testing:
            if model_type == "mrc":
                pred_ids, probabilities = tag_model(input_ids, labels, text_length_list, token_type_ids, 
                                     type_index_in_ids_list, is_training, is_testing)
            else:
                pred_ids, probabilities = tag_model(input_ids, labels, text_length_list, token_type_ids,is_training, is_testing)
        else:
            if model_type == "mrc":
                per_example_loss, loss, pred_ids, probabilities = tag_model(input_ids, labels, text_length_list, 
                                                             token_type_ids, type_index_in_ids_list, is_training)
            else:
                per_example_loss, loss, pred_ids, probabilities = tag_model(input_ids, labels, text_length_list, token_type_ids, is_training)  

        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoints:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoints)
            tf.train.init_from_checkpoint(init_checkpoints, assignment_map)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            print("*" * 15 + "Training" + "*" * 15)
#             train_op = optimization.create_optimizer(loss, args.lr, params["decay_steps"], None, False)
            train_op = optimization.create_optimizer(loss, args.lr, params["train_steps"], params["warmup_steps"], False)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        elif mode == tf.estimator.ModeKeys.EVAL:
            print("*" * 15 + "Evaluation" + "*" * 15)
            pred_ids = tf.argmax(pred_ids, axis=-1)
            pred_ids = tf.one_hot(pred_ids, depth=params["num_labels"], dtype=tf.float32)
            def metric_fn(per_example_loss, label_ids, probabilities):
                logits_split = tf.split(probabilities, params["num_labels"], axis=-1)
                label_ids_split = tf.split(label_ids, params["num_labels"], axis=-1)
                eval_dict = {}
                for j, logits in enumerate(logits_split):
                    label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                    precision, update_op_pre = tf.metrics.precision(label_id_, logits)
                    recall, update_op_recall = tf.metrics.recall(label_id_, logits)
                    f1 = (2 * precision * recall) / (precision + recall)
                    eval_dict[str(j)+'-pre'] = (precision, update_op_pre)
                    eval_dict[str(j)+'-rec'] = (recall, update_op_recall)
                    eval_dict[str(j)+'-f1'] = (f1, tf.identity(f1))
                eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
                eval_dict['label_accuracy'] = tf.metrics.accuracy(labels=tf.argmax(label_ids, axis=-1), 
                                                                  predictions=tf.argmax(probabilities, axis=-1), name='acc_op')

                return eval_dict

            eval_metrics = metric_fn(per_example_loss, labels, pred_ids)
            output_spec = tf.estimator.EstimatorSpec(
                eval_metric_ops=eval_metrics,
                mode=mode,
                loss=loss
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=probabilities
            )
        return output_spec

    return model_fn
# 