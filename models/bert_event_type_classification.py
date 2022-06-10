import tensorflow as tf
import numpy as np
import common_utils
import sys
import math
from bert import modeling, optimization

logger = common_utils.set_logger('Training...')

class BertTypeCLS(object):
    """ BERT多标签二分类模型
    """
    def __init__(self, params, bert_config):
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = params["num_labels"]
        self.class_weight = params["class_weight"]
        self.bert_config = bert_config

    def __call__(self, input_ids, labels, text_length_list, token_type_ids, is_training, is_testing=False):
        """ 模型入口
        @param input_ids: 输入文本的token id， shape: [batch_size, max_length_in_batch]
        @param labels: 标注的标签, shape: [batch_size, num_labels]
        @param text_length_list: 当前batch中文本的实际长度，用于生成mask矩阵
        @param token_type_ids: BERT中的segment ids，0/1，shape: [batch_size, max_length_in_batch]
        """
        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            use_one_hot_embeddings=False, token_type_ids=token_type_ids
        )
        bert_output = bert_model.get_pooled_output() # [batch_size * hidden_size]
        bert_output = tf.layers.dropout(bert_output,rate=self.dropout_rate,training=is_training)
        bert_project = tf.layers.dense(bert_output, self.num_labels) # [batch_size * num_labels]
        pred_prob = tf.nn.sigmoid(bert_project, name="pred_probs")
        if not is_testing:
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=bert_project)
            loss = tf.reduce_mean(per_example_loss * self.class_weight)

            return per_example_loss, loss, pred_prob
        else:
            return pred_prob

class BertTypeCLSModified(object):
    """ MRC多标签分类模型，在原文本后拼接了[unused1] + type1 + [unused2] + [unused1] + type2 + [unused2] ...，引入了标签信息
    """
    def __init__(self, params, bert_config):
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = params["num_labels"]
        self.class_weight = params["class_weight"]
        self.bert_config = bert_config

    def __call__(self, input_ids, labels, text_length_list, token_type_ids, type_index_in_token_ids, is_training,
                 is_testing=False):
        """ 模型入口
        @param input_ids: 输入文本的token id， shape: [batch_size, max_length_in_batch]
        @param labels: 标注的标签, shape: [batch_size, num_labels]
        @param text_length_list: 当前batch中文本的实际长度，用于生成mask矩阵
        @param token_type_ids: BERT中的segment ids，0/1，shape: [batch_size, max_length_in_batch]
        @param type_index_in_token_ids: 用于分类的[unused1]标签在句子中的位置, shape: [batch_size, num_labels]
        """
        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            use_one_hot_embeddings=False, token_type_ids=token_type_ids
        )
        bert_embedding = bert_model.get_sequence_output() # [batch_size, max_length_in_batch, hidden_size]
        bert_embedding = tf.layers.dropout(bert_embedding,rate=self.dropout_rate,training=is_training)
        batch_ids = tf.range(0, tf.shape(bert_embedding)[0])
        batch_ids = tf.expand_dims(batch_ids, 1)
        batch_ids = tf.expand_dims(batch_ids, -1)
        batch_ids = tf.tile(batch_ids, [1, self.num_labels, 1])
        type_index_in_token_ids = tf.expand_dims(type_index_in_token_ids, axis=-1) 
        type_index = tf.concat([batch_ids, type_index_in_token_ids], axis=-1) # [batch_size, num_labels, 2] 对应每个[unused1]标签在矩阵中的位置
        type_head_tensor = tf.gather_nd(bert_embedding, type_index) # 取出所有[unused1]标签对应的向量 [batch_size, num_labels, hidden_size]
        bert_project = tf.layers.dense(type_head_tensor, 1)

        pred_prob = tf.nn.sigmoid(bert_project, name="pred_probs")
        pred_prob = tf.squeeze(pred_prob, axis=-1)

        if not is_testing:
            labels = tf.expand_dims(labels, axis=-1)
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=bert_project)
            loss = tf.reduce_mean(per_example_loss * self.class_weight)

            return per_example_loss, loss, pred_prob
        else:
            return pred_prob

        
class BertTypeCLSAdaptive(object):
    """ CasEE多标签分类模型https://aclanthology.org/2021.findings-acl.14.pdf
    """
    def __init__(self, params, bert_config):
        rng = np.random.RandomState(3435)
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = params["num_labels"]
        self.class_weight = params["class_weight"]
        self.bert_config = bert_config
        self.hidden_size = bert_config.hidden_size
        evt_emb = np.asarray(0.01 * rng.standard_normal(size=(self.num_labels ,self.hidden_size)),dtype=np.float32)
        self.type_emb = tf.Variable(evt_emb,name='type_emb') # 类型向量矩阵
    
    def __call__(self, input_ids, labels, text_length_list, token_type_ids, is_training, is_testing=False):
        """ 模型入口
        @param input_ids: 输入文本的token id， shape: [batch_size, max_length_in_batch]
        @param labels: 标注的标签, shape: [batch_size, num_labels]
        @param text_length_list: 当前batch中文本的实际长度，用于生成mask矩阵
        @param token_type_ids: BERT中的segment ids，0/1，shape: [batch_size, max_length_in_batch]
        """
        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            use_one_hot_embeddings=False, token_type_ids=token_type_ids
        )
        context = bert_model.get_sequence_output() # [batch_size, max_length_in_batch, hidden_size]
        context = tf.expand_dims(context, 1)
        context = tf.tile(context, (1, self.num_labels, 1, 1)) # [batch_size, num_labels, max_length_in_batch, hidden_size]
        type_emb = tf.expand_dims(self.type_emb, 0)
        type_emb = tf.expand_dims(type_emb, 2)
        type_emb = tf.broadcast_to(type_emb, tf.shape(context)) # [batch_size, num_labels, max_length_in_batch, hidden_size]
        concat_hidden = tf.concat([type_emb, context, tf.abs(type_emb - context), type_emb * context], axis=-1) 
        # [batch_size, num_labels, max_length_in_batch, 4 * hidden_size]
        
        project_hidden = tf.layers.dense(concat_hidden, self.hidden_size * 4, name='dense1')
        project_hidden = tf.tanh(project_hidden)
        scores = tf.layers.dense(project_hidden, 1, name='dense2') # [batch_size, num_labels, max_length_in_batch, 1]
        scores = tf.layers.dropout(scores,rate=self.dropout_rate, training=is_training)
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
        g = tf.matmul(scores, context) # [batch_size, num_labels, 1, hidden_size]，通过attention得到每个类型各自对应的动态句子表示
        g = tf.squeeze(g, 2)
        type_emb_2 = tf.expand_dims(self.type_emb, 0)
        type_emb_2 = tf.broadcast_to(type_emb_2, tf.shape(g)) # [batch_size, num_labels, hidden_size] 再和事件类型向量做一次交互
        concat_hidden_2 = tf.concat([type_emb_2, g, tf.abs(type_emb_2 - g), type_emb_2 * g], axis=-1)
        project_hidden_2 = tf.layers.dense(concat_hidden_2, self.hidden_size * 4, name='dense1', reuse=True) # [batch_size, num_labels, 4 * hidden_size]
        project_hidden_2 = tf.tanh(project_hidden_2)
        context_type_awared = tf.layers.dense(project_hidden_2, 1, name='dense2', reuse=True) 
        context_type_awared = tf.squeeze(context_type_awared, -1) # [batch_size, num_labels] 
        pred_prob = tf.nn.sigmoid(context_type_awared, name="pred_probs")
        if not is_testing:
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=context_type_awared)
            loss = tf.reduce_mean(per_example_loss * self.class_weight)

            return per_example_loss, loss, pred_prob
        else:
            return pred_prob
        

def bert_classification_model_fn_builder(bert_config_file, init_checkpoints, args):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        if isinstance(features, dict):
            features = features['words'], features['token_type_ids'], features['text_length'], features[
                'type_index_in_ids_list']
        input_ids, token_type_ids, text_length_list, type_index_in_ids_list = features
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_testing = (mode == tf.estimator.ModeKeys.PREDICT)
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        model_type = params["model_type"]
        assert model_type in ["mrc", "casee", "bert"], "Undefined model type {}".format(model_type)
        if model_type == "mrc":
            tag_model = BertTypeCLSModified(params, bert_config)  
        elif model_type == "casee": 
            tag_model = BertTypeCLSAdaptive(params,bert_config)
        else:
            tag_model = BertTypeCLS(params,bert_config)
        if is_testing:
            if model_type == "mrc":
                pred_ids = tag_model(input_ids, labels, text_length_list, token_type_ids, 
                                     type_index_in_ids_list, is_training, is_testing)
            else:
                pred_ids = tag_model(input_ids, labels, text_length_list, token_type_ids,is_training, is_testing)
        else:
            if model_type == "mrc":
                per_example_loss, loss, pred_ids = tag_model(input_ids, labels, text_length_list, 
                                                             token_type_ids, type_index_in_ids_list, is_training)
            else:
                per_example_loss, loss, pred_ids = tag_model(input_ids, labels, text_length_list, token_type_ids, is_training)
        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoints:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoints)
            tf.train.init_from_checkpoint(init_checkpoints, assignment_map)
        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss, args.lr, params["train_steps"], params["warmup_steps"], False)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        elif mode == tf.estimator.ModeKeys.EVAL:

            pred_ids = tf.where(pred_ids > 0.5, tf.ones_like(pred_ids), tf.zeros_like(pred_ids))

            def metric_fn(per_example_loss, label_ids, probabilities):

                logits_split = tf.split(probabilities, params["num_labels"], axis=-1)
                label_ids_split = tf.split(label_ids, params["num_labels"], axis=-1)
                eval_dict = {}
                for j, logits in enumerate(logits_split):
                    label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                    precision, update_op_pre = tf.metrics.precision(label_id_, logits)
                    recall, update_op_recall = tf.metrics.recall(label_id_, logits)
                    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
                    eval_dict[str(j)+'-pre'] = (precision, update_op_pre)
                    eval_dict[str(j)+'-rec'] = (recall, update_op_recall)
                    eval_dict[str(j)+'-f1'] = (f1, tf.identity(f1))
                eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
                eval_dict['label_accuracy'] = tf.metrics.accuracy(labels=label_ids, predictions=probabilities, name='acc_op')

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
                predictions=pred_ids
            )
        return output_spec

    return model_fn

