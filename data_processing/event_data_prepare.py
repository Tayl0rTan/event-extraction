import codecs
import json
import tensorflow as tf
import random
import numpy as np
import re
import os
from bert4keras.tokenizers import Tokenizer


class ClassifyDataPrepare:
    def __init__(self, config):
        vocab_file_path = os.path.join(config.get("bert_pretrained_model_path"), config.get("vocab_file"))
        types_file =  os.path.join(config.get("data_dir"), config.get("type_file"))
        self.max_seq_length = config.get("max_seq_length")
        self.labels_map, self.id2labels_map = self.read_slot(types_file)
        self.labels_map_len = len(self.labels_map)
        self.tokenizer = Tokenizer(vocab_file_path, do_lower_case=True)
        self.config = config

    def read_slot(self, slot_file):
        labels_map = {}
        id2labels_map = {}
        with codecs.open(slot_file, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip()
                line = line.strip("\n")
                arr = line.split("\t")
                print(arr)
                labels_map[arr[0]] = int(arr[1])
                id2labels_map[int(arr[1])] = arr[0]
        return labels_map, id2labels_map

    def truncate_seq_head_tail(self, tokens, head_len, tail_len, max_length):
        if len(tokens) <= max_length:
            return tokens
        else:
            head_tokens = tokens[0: head_len]
            tail_tokens = tokens[len(tokens) - tail_len:]
            return head_tokens + tail_tokens
    
    def gen_npy_data(self, input_file, set_name='train'):
        input_data = json.load(open(input_file))
        data_list, token_type_id_list, label_list, type_index_in_token_ids_list = self.parse_data_from_json(input_data)
        data_dir = self.config.get("data_dir")
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        np.save("{}/token_ids_{}.npy".format(data_dir, set_name), data_list)
        np.save("{}/labels_{}.npy".format(data_dir, set_name), label_list)
        np.save("{}/token_type_ids_{}.npy".format(data_dir, set_name), token_type_id_list)
        np.save("{}/type_index_in_token_ids_{}.npy".format(data_dir, set_name), type_index_in_token_ids_list)

    def parse_data_from_json(self, input_data):
        data_list = []
        label_list = []
        token_type_id_list = []
        type_index_in_token_ids_list = []
        all_intent_type_split = []
        mrc_data = self.config.get('model_type').lower() == 'mrc'
        for i in range(self.config.get("label_nums")):
            intent_type_str = self.id2labels_map.get(i)
            all_intent_type_split.append(intent_type_str)

        for data in input_data:
            sentence = data["text"]
            if not sentence and 'raw_text' in data:
                sentence = data['raw_text']
            intent_list = data["event_list"]
            intent_type_label_list = [0] * self.labels_map_len
            type_index_in_token_ids = []
            text_len_for_intent_raw_str = 0
            for intent in intent_list:
                intent_type = intent["event_type"]
#                 print(intent_type, self.labels_map)
                index_of_intent_type = self.labels_map[intent_type]
                intent_type_label_list[index_of_intent_type] = 1
            token_ids_org, token_type_ids = self.tokenizer.encode(sentence)
            token_ids = token_ids_org
            type_token_len = 0
            suffix_token_ids = []

            for index, intent_type_raw in enumerate(all_intent_type_split):
                intent_type_token_ids = self.tokenizer.encode(intent_type_raw)[0]
                text_len_for_intent_raw_str += len(intent_type_token_ids)

            text_allow_len = 510 - text_len_for_intent_raw_str if mrc_data else 510
            if len(token_ids) > text_allow_len:
                # head + tail truncate
                header_len = int(text_allow_len / 4)
                tail_len = text_allow_len - header_len
                dealt_token_ids = token_ids[1:-1]
                prefix_token_ids = self.truncate_seq_head_tail(dealt_token_ids, header_len, tail_len, text_allow_len)
                token_ids = [token_ids_org[0]] + prefix_token_ids + [token_ids_org[-1]]
                token_type_ids = [0] * len(token_ids)
            if mrc_data:
                for index, intent_type_raw in enumerate(all_intent_type_split):
                    type_index_in_token_ids.append(len(token_ids))
                    intent_type_token_ids = self.tokenizer.encode(intent_type_raw)[0]
                    intent_type_token_ids[0] = 1
                    intent_type_token_ids[-1] = 2
                    type_token_len += len(intent_type_token_ids)
                    suffix_token_ids.extend(intent_type_token_ids)
                    token_ids = token_ids + intent_type_token_ids
                    token_type_ids.extend([1] * len(intent_type_token_ids))
            else:
                type_index_in_token_ids = [0] * self.config.get("label_nums") # not unused actually
            data_list.append(token_ids)
            token_type_id_list.append(token_type_ids)
            label_list.append(intent_type_label_list)
            type_index_in_token_ids_list.append(type_index_in_token_ids)


        return data_list, token_type_id_list, label_list, type_index_in_token_ids_list

    def trans_single_data_for_test(self, text):
        token_ids_org, token_type_ids = self.tokenizer.encode(text)
        all_intent_type_split = []
        for i in range(self.config.get("label_nums")):
            intent_type_str = self.id2labels_map.get(i)
            all_intent_type_split.append(intent_type_str)
        token_ids = token_ids_org
        type_token_len = 0
        text_len_for_intent_raw_str = 0
        type_index_in_token_ids = []
        for index, intent_type_raw in enumerate(all_intent_type_split):
            intent_type_token_ids = self.tokenizer.encode(intent_type_raw)[0]
            text_len_for_intent_raw_str += len(intent_type_token_ids)
        text_allow_len = 510 - text_len_for_intent_raw_str
        if len(token_ids) > text_allow_len:
            header_len = int(text_allow_len / 4)
            tail_len = text_allow_len - header_len
            dealt_token_ids = token_ids[1:-1]
            prefix_token_ids = self.truncate_seq_head_tail(dealt_token_ids, header_len, tail_len, text_allow_len)
            token_ids = [token_ids_org[0]] + prefix_token_ids + [token_ids_org[1]]
            token_type_ids = [0] * len(token_ids)
        if self.config.get('model_type').lower() == 'mrc':
            for index, intent_type_raw in enumerate(all_intent_type_split):
                type_index_in_token_ids.append(len(token_ids))
                # record the unused positions for classification
                intent_type_token_ids = self.tokenizer.encode(intent_type_raw)[0]
                intent_type_token_ids[0] = 1
                intent_type_token_ids[-1] = 2
                type_token_len += len(intent_type_token_ids)
                token_ids = token_ids + intent_type_token_ids
                token_type_ids.extend([1] * len(intent_type_token_ids))
        else:
            type_index_in_token_ids = [0] * self.config.get("label_nums")
        return token_ids, token_type_ids, type_index_in_token_ids


class EventRolePrepareMRC:
    def __init__(self, config):
        
        vocab_file_path = os.path.join(config.get("bert_pretrained_model_path"), config.get("vocab_file"))
        # schema file path
        schema_file = os.path.join(config.get("data_dir"), config.get("event_schema"))
        self.max_seq_length = config.get("max_seq_length")
        self.schema_dict = self.parse_schema_type(schema_file)
        self.tokenizer = Tokenizer(vocab_file_path, do_lower_case=True)
        self.config = config

    def gen_npy_data(self, input_file, set_name='train'):
        input_data = json.load(open(input_file))
        data_list, label_start_list, label_end_list, query_len_list, token_type_id_list = self.parse_data_from_json(input_data)
        data_dir = self.config.get("data_dir")
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        np.save("{}/token_ids_{}.npy".format(data_dir, set_name), data_list)
        np.save("{}/labels_start_{}.npy".format(data_dir, set_name), label_start_list)
        np.save("{}/labels_end_{}.npy".format(data_dir, set_name), label_end_list)
        np.save("{}/token_type_ids_{}.npy".format(data_dir, set_name), token_type_id_list)
        np.save("{}/query_lens_{}.npy".format(data_dir, set_name), query_len_list)


    def parse_schema_type(self, schema_file):
        schema_event_dict = {}
        with codecs.open(schema_file, 'r', 'utf-8') as fr:
            for line in fr:
                s_json = json.loads(line)
                role_list = s_json.get("role_list")
                schema_event_dict.update({s_json.get("event_type"): [ele.get("role") for ele in role_list]})
        return schema_event_dict

    def tranform_singlg_data_example(self, sent, roles_list):
        words = list(sent)
        sent_ori_2_new_index = {}
        new_words = []
        new_start, new_end = -1, -1
        new_roles_list = {}
        for role_type, role in roles_list.items():
            new_roles_list[role_type] = {
                "role_type": role_type,
                "start": -1,
                "end": -1
            }

        for i, w in enumerate(words):
            for role_type, role in roles_list.items():
                if i == role["start"]:
                    new_roles_list[role_type]["start"] = len(new_words)
                if i == role["end"]:
                    new_roles_list[role_type]["end"] = len(new_words)

            if len(w.strip()) == 0:
                sent_ori_2_new_index[i] = -1
                for role_type, role in roles_list.items():
                    if i == role["start"]:
                        new_roles_list[role_type]["start"] += 1
                    if i == role["end"]:
                        new_roles_list[role_type]["end"] -= 1
            else:
                sent_ori_2_new_index[i] = len(new_words)
                new_words.append(w)
        for role_type, role in new_roles_list.items():
            if role["start"] > -1:
                role["text"] = u"".join(
                    new_words[role["start"]:role["end"] + 1])
            if role["end"] == len(new_words):
                role["end"] = len(new_words) - 1

        return [words, new_words, sent_ori_2_new_index, new_roles_list]

    def gen_query_for_each_sample(self, event_type, role_type):
        query_str_final = "找到{}事件中的{}".format(event_type, role_type)
        return query_str_final

    def parse_data_from_json(self, input_data):
        data_list = []
        label_start_list = []
        label_end_list = []
        query_len_list = []
        token_type_id_list = []
        for index in range(len(input_data)):
            data = input_data[index]
            sentence = data["text"] + '-'
            sentence_token_ids, sentence_token_type_ids = self.tokenizer.encode(sentence)
            sentence_token_type_ids = [ids + 1 for ids in sentence_token_type_ids]
            if len(sentence_token_ids) != len(sentence_token_type_ids):
                print(sentence)
            event_list = data["event_list"]
            for event in event_list:
                dealt_role_list = []
                event_type = event["event_type"]
                roles_list = {}
                role_type_dict = {}
                for index, role in enumerate(event["arguments"]):
                    role_type = role["role"]
                    if role_type in role_type_dict:
                        role_type_dict.get(role_type).append(index)
                    else:
                        role_type_dict.update({role_type: [index]})
                for role_type_key, argument_index_list in role_type_dict.items():

                    dealt_role_list.append(role_type_key)
                    query_word = self.gen_query_for_each_sample(event_type, role_type_key)
                    query_word_token_ids, query_word_token_type_ids = self.tokenizer.encode(query_word)
                    cur_start_labels = [0] * len(sentence_token_ids)
                    cur_end_labels = [0] * len(sentence_token_ids)
                    for argument_index in argument_index_list:

                        role_argument = event["arguments"][argument_index]
                        role_text = role_argument["argument"]
                        a_token_ids = self.tokenizer.encode(role_text)[0][1:-1]
                        start_index = search(a_token_ids, sentence_token_ids)
                        role_end = start_index + len(a_token_ids) - 1
                        # binary class
                        if start_index != -1:
                            cur_start_labels[start_index] = 1
                            cur_end_labels[role_end] = 1

                    cur_final_token_ids = query_word_token_ids + sentence_token_ids[1:]
                    cur_final_token_type_ids = query_word_token_type_ids + sentence_token_type_ids[1:]
                    cur_start_labels = [0] * len(query_word_token_ids) + cur_start_labels[1:]
                    cur_end_labels = [0] * len(query_word_token_ids) + cur_end_labels[1:]

                    data_list.append(cur_final_token_ids)
                    label_start_list.append(cur_start_labels)
                    label_end_list.append(cur_end_labels)
                    query_len_list.append(len(query_word_token_ids))
                    token_type_id_list.append(cur_final_token_type_ids)

                schema_role_list = self.schema_dict.get(event_type)
                for schema_role in schema_role_list:
                    if schema_role not in dealt_role_list:
                        query_word = self.gen_query_for_each_sample(event_type, schema_role)
                        query_word_token_ids, query_word_token_type_ids = self.tokenizer.encode(query_word)
                        cur_final_token_ids = query_word_token_ids + sentence_token_ids[1:]
                        cur_final_token_type_ids = query_word_token_type_ids + sentence_token_type_ids[1:]
                        cur_start_labels = [0] * len(cur_final_token_ids)
                        cur_end_labels = [0] * len(cur_final_token_ids)
                        data_list.append(cur_final_token_ids)
                        label_start_list.append(cur_start_labels)
                        label_end_list.append(cur_end_labels)
                        query_len_list.append(len(query_word_token_ids))
                        token_type_id_list.append(cur_final_token_type_ids)

        return data_list, label_start_list, label_end_list, query_len_list, token_type_id_list


    def trans_single_data_for_test(self, text, query_word, max_seq_len):
        text = text + '-'
        tokens = self.tokenizer.tokenize(text)
        query_token_ids, query_token_type_ids = self.tokenizer.encode(query_word)
        query_len = len(query_token_ids)

        mapping = self.tokenizer.rematch(text, tokens)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        while len(token_ids) - 1 > max_seq_len - query_len:
            token_ids.pop(-2)
        token_type_ids = [1] * len(token_ids)
        final_token_ids = query_token_ids + token_ids[1:]
        final_token_type_ids = query_token_type_ids + token_type_ids[1:]

        return final_token_ids, len(query_token_ids), final_token_type_ids, mapping

    
class MultiTaskClassifyDataPrepare:
    def __init__(self, config):
        vocab_file_path = os.path.join(config.get("bert_pretrained_model_path"), config.get("vocab_file"))
        types_file =  os.path.join(config.get("data_dir"), config.get("type_file"))
        self.max_seq_length = config.get("max_seq_length")
        self.labels_map, self.id2labels_map, self.label_tid2type_map, self.label_type2tid_map = self.read_slot(types_file)
        
        self.label_tid2len_list = [len(self.labels_map[self.label_tid2type_map[label_tid]]) for label_tid in  range(len(self.labels_map))]#[4,5,6]每个位置对应该位置id的标签类别下的数目，比如Type_0有4个标签，Type_1有5个标签，……
        
        self.tokenizer = Tokenizer(vocab_file_path, do_lower_case=True)
        self.config = config

    
    def read_slot(self, slot_file):
        labels_map = {}
        id2labels_map = {}
        label_tid2type_map={}
        label_type2tid_map={}
        with codecs.open(slot_file, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip()
                line = line.strip("\n")
                arr = line.split("\t")
                print(arr)
                label, idx, label_type, label_tid=arr[0], int(arr[1]),arr[2], int(arr[3])
                
                labels_map[label_type]=labels_map.get(label_type,{})
                labels_map[label_type][label] = idx
                
                id2labels_map[label_type]=id2labels_map.get(label_type,{})
                id2labels_map[label_type][idx] = label
                
                assert (label_tid not in label_tid2type_map) or (label_tid2type_map[label_tid]==label_type)#每个label_type的index必须一致，防止写错
                label_tid2type_map[label_tid]=label_type
        
        label_type2tid_map={j:i for i,j in label_tid2type_map.items()}
        return labels_map, id2labels_map,label_tid2type_map, label_type2tid_map

    def truncate_seq_head_tail(self, tokens, head_len, tail_len, max_length):
        if len(tokens) <= max_length:
            return tokens
        else:
            head_tokens = tokens[0: head_len]
            tail_tokens = tokens[len(tokens) - tail_len:]
            return head_tokens + tail_tokens
    
    def gen_npy_data(self, input_file, set_name='train'):
        input_data = json.load(open(input_file))
        data_list, token_type_id_list, label_list = self.parse_data_from_json(input_data)
        data_dir = self.config.get("data_dir")
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        np.save("{}/token_ids_{}.npy".format(data_dir, set_name), data_list)
        for label_type in label_list:
            np.save("{}/labels_{}_{}.npy".format(data_dir, label_type, set_name), label_list[label_type])
        np.save("{}/token_type_ids_{}.npy".format(data_dir, set_name), token_type_id_list)

    def parse_data_from_json(self, input_data):
        data_list = []
        label_list = {label_type:[] for label_type in self.label_type2tid_map}#{label_type:[]}
        token_type_id_list = []
        type_index_in_token_ids_list = []

        for data in input_data:
            sentence = data["text"]
            if not sentence and 'raw_text' in data:
                sentence = data['raw_text']
            intent_list = data["event_list"]
            
            intent_type_label_list = {label_type:[0] * self.label_tid2len_list[label_tid]  for label_type,label_tid in self.label_type2tid_map.items()}
            type_index_in_token_ids = []
            text_len_for_intent_raw_str = 0
            for intent in intent_list:#intent={'type_1': 'aaa'}
                label_type=list(intent.keys())[0]
                intent_type = intent[label_type]
#                 print(intent_type, self.labels_map)
                index_of_intent_type = self.labels_map[label_type][intent_type]
                intent_type_label_list[label_type][index_of_intent_type] = 1
            token_ids_org, token_type_ids = self.tokenizer.encode(sentence)
            token_ids = token_ids_org

            text_allow_len = 510
            if len(token_ids) > text_allow_len:
                # head + tail truncate
                header_len = int(text_allow_len / 4)
                tail_len = text_allow_len - header_len
                dealt_token_ids = token_ids[1:-1]
                prefix_token_ids = self.truncate_seq_head_tail(dealt_token_ids, header_len, tail_len, text_allow_len)
                token_ids = [token_ids_org[0]] + prefix_token_ids + [token_ids_org[-1]]
                token_type_ids = [0] * len(token_ids)
            
            data_list.append(token_ids)
            token_type_id_list.append(token_type_ids)
            for label_type in label_list:
                label_list[label_type].append(intent_type_label_list[label_type])

        return data_list, token_type_id_list, label_list

    def trans_single_data_for_test(self, text):
        token_ids_org, token_type_ids = self.tokenizer.encode(text)
        token_ids = token_ids_org

        text_allow_len = 510
        if len(token_ids) > text_allow_len:
            header_len = int(text_allow_len / 4)
            tail_len = text_allow_len - header_len
            dealt_token_ids = token_ids[1:-1]
            prefix_token_ids = self.truncate_seq_head_tail(dealt_token_ids, header_len, tail_len, text_allow_len)
            token_ids = [token_ids_org[0]] + prefix_token_ids + [token_ids_org[1]]
            token_type_ids = [0] * len(token_ids)
        
        return token_ids, token_type_ids
    
    
def event_index_data_generator_bert_class(input_Xs, token_type_ids, type_index_ids_list, labels):
    for index in range(len(input_Xs)):
        input_x = input_Xs[index]
        label = labels[index]
        token_type_id = token_type_ids[index]
        type_index_ids = type_index_ids_list[index]
        yield (input_x, token_type_id, len(input_x), type_index_ids), label


def event_index_class_input_bert_fn(input_Xs, token_type_ids, type_index_ids_list, label_map_len, is_training,
                                    is_testing, args, input_Ys=None):
    _shapes = (([None], [None], (), [label_map_len]), [None])
    _types = ((tf.int32, tf.int32, tf.int32, tf.int32), tf.float32)
    _pads = ((0, 0, 0, 0), 0.0)
    ds = tf.data.Dataset.from_generator(
        lambda: event_index_data_generator_bert_class(input_Xs, token_type_ids, type_index_ids_list, input_Ys),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        ds = ds.shuffle(args.shuffle_buffer).repeat(args.epochs)
        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads)
    else:
        if is_testing:
            ds = ds.padded_batch(args.test_batch_size, _shapes, _pads)
        else:
            ds = ds.padded_batch(args.valid_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds


def event_data_generator_bert_mrc(input_Xs, start_Ys, end_Ys, token_type_ids, query_lens):
    for index in range(len(input_Xs)):
        input_x = input_Xs[index]
        start_y = start_Ys[index]
        end_y = end_Ys[index]
        token_type_id = token_type_ids[index]
        query_len = query_lens[index]
        yield (input_x, len(input_x), query_len, token_type_id), (start_y, end_y)


def event_input_bert_mrc_fn(input_Xs, start_Ys, end_Ys, token_type_ids, query_lens, is_training, is_testing, args):
    _shapes = (([None], (), (), [None]), ([None], [None]))
    _types = ((tf.int32, tf.int32, tf.int32, tf.int32), (tf.int32, tf.int32))
    _pads = ((0, 0, 0, 0), (0, 0))
    ds = tf.data.Dataset.from_generator(
        lambda: event_data_generator_bert_mrc(input_Xs, start_Ys, end_Ys, token_type_ids, query_lens),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        ds = ds.shuffle(args.shuffle_buffer).repeat(args.epochs)
    if is_training:
        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads, )
    else:
        if is_testing:
            ds = ds.padded_batch(args.test_batch_size, _shapes, _pads)
        else:
            ds = ds.padded_batch(args.valid_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds

def event_multi_task_generator_bert_class(input_Xs, token_type_ids, labels):
    for index in range(len(input_Xs)):
        input_x = input_Xs[index]
        label = tuple([l[index] for l in labels])
        token_type_id = token_type_ids[index]
        yield (input_x, token_type_id, len(input_x)), label

def event_multi_task_input_bert_fn(input_Xs, token_type_ids, is_training,
                                    is_testing, args, input_Ys=None, label_num=1):
    _shapes = (([None], [None], ()), tuple([[None] for _ in range(label_num)]))
    _types = ((tf.int32, tf.int32, tf.int32), tuple([tf.float32 for _ in range(label_num)]))
    _pads = ((0, 0, 0), tuple([0.0 for _ in range(label_num)]))
    ds = tf.data.Dataset.from_generator(
        lambda: event_multi_task_generator_bert_class(input_Xs, token_type_ids, input_Ys),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        ds = ds.shuffle(args.shuffle_buffer).repeat(args.epochs)
        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads)
    else:
        if is_testing:
            ds = ds.padded_batch(args.test_batch_size, _shapes, _pads)
        else:
            ds = ds.padded_batch(args.valid_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    else:
        print('---search failed-----')
        return -1