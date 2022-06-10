import tensorflow as tf
import re
import os
import json
import numpy as np
import codecs
import logging

from event_config import type_config, status_config, role_config
from copy import deepcopy
from data_processing.event_data_prepare import ClassifyDataPrepare, EventRolePrepareMRC, MultiTaskClassifyDataPrepare
from tensorflow.contrib import predictor
from tqdm import tqdm

from argparse import ArgumentParser

#NOSTATUS = {'餐损', '商品售完'}
NOSTATUS = {}

class TypePredictor:
    def __init__(self, config):
        self.data_loader = ClassifyDataPrepare(config)
        self.predict_fn = predictor.from_saved_model(config.get('best_model_pb'))
        self.model_type = config.get('model_type')
        self.config = config
        logging.info('-'*15 + 'Loaded Type Model: ' + self.model_type + '-'*15)

    def predict_single_sample(self, text):
        words_input, token_type_ids, type_index_in_token_ids = self.data_loader.trans_single_data_for_test(
            text)
        predictions = self.predict_fn({'words': [words_input], 'text_length': [
                len(words_input)], 'token_type_ids': [token_type_ids], 'type_index_in_ids_list': [type_index_in_token_ids]})
        label = predictions["output"][0]
        return label

    def predict_event_type(self, text_list, threshold=0.5):
        predict_result = []
        logging.info('-'*15 + 'Predicting event types' + '-'*15)
        for text in tqdm(text_list):
            if isinstance(text, dict):
                text = text['text']
            sample = {'text': text, 'event_list': []}
            label_output = self.predict_single_sample(text)
            one_hot_output = np.argwhere(label_output > threshold)
            sample['event_list'] = [{'event_type': self.data_loader.id2labels_map.get(
                ele[0])} for ele in one_hot_output]
            predict_result.append(sample)
        return predict_result  

    
class StatusPredictor:
    def __init__(self, config):
        self.data_loader = ClassifyDataPrepare(config)
        self.predict_fn = predictor.from_saved_model(config.get('best_model_pb'))
        self.model_type = config.get('model_type')
        self.config = config
        logging.info('-'*15 + 'Loaded Status Model: ' + self.model_type + '-'*15)

    def predict_single_sample(self, text):
        words_input, token_type_ids, type_index_in_token_ids = self.data_loader.trans_single_data_for_test(
            text)
        predictions = self.predict_fn({'words': [words_input], 'text_length': [
                len(words_input)], 'token_type_ids': [token_type_ids], 'type_index_in_ids_list': [type_index_in_token_ids]})
        label = predictions["output"][0]
        return label

    def predict_status(self, sample_list):
        predict_result = []
        logging.info('-'*15 + 'Predicting event status' + '-'*15)
        for sample in tqdm(sample_list):
            event_list = sample['event_list']
            text = sample['text']
            for event in event_list:
                evt_type = event['event_type']
                if evt_type not in NOSTATUS:
                    text_with_type = '{}事件：{}'.format(evt_type, text)
                    label_output = self.predict_single_sample(text_with_type)
                    idx = np.argmax(label_output)
                    score=np.max(label_output)
                    event['status'] = {'label':self.data_loader.id2labels_map.get(idx), 'score':score.tolist()}
        return sample_list
    

class RolePredictor:
    def __init__(self, config):
        self.data_loader = EventRolePrepareMRC(config)
        self.predict_fn = predictor.from_saved_model(config.get('best_model_pb'))
        self.config = config
        logging.info('-'*15 + 'Loaded Role Model Successfully! ' + '-'*15)

    def predict_single_sample(self, token_ids, query_len, token_type_ids):
        text_length = len(token_ids)
        predictions = self.predict_fn({'words': [token_ids], 'text_length': [text_length], 'query_length': [query_len], 'token_type_ids': [token_type_ids]})
        start_ids, end_ids = predictions.get("start_ids")[0], predictions.get("end_ids")[0]
        return start_ids, end_ids

    def predict_role(self, sample_list):
        predict_sample_list = deepcopy(sample_list)
        logging.info('-'*15 + 'Predicting event roles' + '-'*15)
        for sample in tqdm(predict_sample_list):
            event_list = sample['event_list']
            text = sample['text']
            if not event_list:
                continue
            for event in event_list:
                event_arguments = []
                cur_event_type = event['event_type']
                corresponding_role_type_list = self.data_loader.schema_dict.get(cur_event_type)
                if not corresponding_role_type_list: continue
                for index, cur_role_type in enumerate(corresponding_role_type_list):
                    cur_query_word = self.data_loader.gen_query_for_each_sample(cur_event_type, cur_role_type)
                    token_ids, query_len, token_type_ids, token_mapping = self.data_loader.trans_single_data_for_test(
                        text, cur_query_word, 512)
                    start_ids, end_ids = self.predict_single_sample(token_ids, query_len, token_type_ids)
                    token_mapping = token_mapping[1: -1]
                    start_ids, end_ids = start_ids[query_len:-1], end_ids[query_len:-1]
                    entity_list = extract_entity_from_start_end_ids(text, start_ids, end_ids, token_mapping)
                    if len(entity_list) != 0:
                        # 暂时的方案，后续考虑如果识别出多个论元候选，按照概率高低筛选出最佳候选
                        entity = entity_list[0]
                        event_arguments.append({"role": cur_role_type, "argument": entity})
                event["arguments"] = event_arguments
        return predict_sample_list
    
class MultiTaskPredictor:
    def __init__(self, config):
        self.data_loader = MultiTaskClassifyDataPrepare(config)
        self.predict_fn = predictor.from_saved_model(config.get('best_model_pb'))
        self.model_type = config.get('model_type')
        self.config = config
        logging.info('-'*15 + 'Loaded Status Model: ' + self.model_type + '-'*15)

    def predict_single_sample(self, text):
        words_input, token_type_ids = self.data_loader.trans_single_data_for_test(text)
        predictions = self.predict_fn({'words': [words_input], 'text_length': [
                len(words_input)], 'token_type_ids': [token_type_ids]})
        label = predictions["output"][0]
        return label

    def predict_multi_task(self, sample_list):
        predict_result = []
        logging.info('-'*15 + 'Predicting event multi_task' + '-'*15)
        for sample in tqdm(sample_list):
            event_list = sample['event_list']
            text = sample['text']
            for event in event_list:
                evt_type = event['event_type']
                if True or evt_type not in NOSTATUS:
                    text_with_type = '{}事件：{}'.format(evt_type, text)
                    event['multi_task']={}
                    label_output = self.predict_single_sample(text_with_type)
                    class_split_idx=[sum(self.data_loader.label_tid2len_list[:tid+1]) for tid in range(len(self.data_loader.label_tid2len_list))]
                    predictions_list=np.split(label_output, class_split_idx, axis=-1)[:-1]
                    pred_ids_list = [np.argmax(predictions, axis=-1) for predictions in predictions_list]
                    pred_scores_list = [np.max(predictions, axis=-1) for predictions in predictions_list]
                    for label_tid, (pred_id, pred_score) in enumerate(zip(pred_ids_list, pred_scores_list)):
                        label_type=self.data_loader.label_tid2type_map[label_tid]
                        labels_map=self.data_loader.id2labels_map[label_type]
                        label=labels_map[pred_id]
                        event['multi_task'][label_type]={'label':label,'score':pred_score.tolist()}
        return sample_list
    
    
class PredictPipeline:
    def __init__(self):
        self.type_predictor = TypePredictor(type_config)
        self.stat_predictor = StatusPredictor(status_config)
        self.role_predictor = RolePredictor(role_config)
    
    def predict(self, text_list):
        text_with_types = self.type_predictor.predict_event_type(text_list)
        text_with_stats = self.stat_predictor.predict_status(text_with_types)
        text_with_all = self.role_predictor.predict_role(text_with_stats)
        return text_with_all
    
    
def extract_entity_from_start_end_ids(text, start_ids, end_ids, token_mapping):
    """根据开始，结尾标识，找到对应的实体"""
    entity_list = []
    for i, start_id in enumerate(start_ids):
        if start_id == 0:
            continue
        if end_ids[i] == 1:
            # start和end相同
            entity_str = "".join([text[char_index]
                                  for char_index in token_mapping[i]])
            entity_list.append(entity_str)
            continue
        j = i + 1
        find_end_tag = False
        while j < len(end_ids):
            # 若在遇到end=1之前遇到了新的start=1,则停止该实体的搜索
            if start_ids[j] == 1:
                break
            if end_ids[j] == 1:
                entity_str_index_list = []
                for index in range(i, j + 1):
                    entity_str_index_list.extend(token_mapping[index])

                entity_str = "".join([text[char_index] for char_index in entity_str_index_list])
                entity_list.append(entity_str)
                find_end_tag = True
                break
            else:
                j += 1
        if not find_end_tag:
            entity_str = "".join([text[char_index]
                                  for char_index in token_mapping[i]])
            entity_list.append(entity_str)
    return entity_list
