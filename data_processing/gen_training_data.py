import os
import sys
from argparse import ArgumentParser
from data_processing.event_data_prepare import ClassifyDataPrepare, EventRolePrepareMRC,MultiTaskClassifyDataPrepare
from event_config import type_config, status_config, role_config, multi_task_config
    
def generate_type_data():
    """
    generate type classification data
    """
    data_loader = ClassifyDataPrepare(type_config)  
    data_dir = type_config.get("data_dir")
    for file_str in ['train', 'dev', 'test']:
        data_file = os.path.join(data_dir, type_config.get("{}_data".format(file_str)))
        data_loader.gen_npy_data(data_file, file_str)

def generate_status_data():
    data_loader = ClassifyDataPrepare(status_config)  
    data_dir = status_config.get("data_dir")
    for file_str in ['train', 'dev', 'test']:
        data_file = os.path.join(data_dir, status_config.get("{}_data".format(file_str)))
        data_loader.gen_npy_data(data_file, file_str)

def generate_role_data():
    data_loader = EventRolePrepareMRC(role_config)
    data_dir = role_config.get("data_dir")
    for file_str in ['train', 'dev']:
        data_file = os.path.join(data_dir, role_config.get("{}_data".format(file_str)))
        data_loader.gen_npy_data(data_file, file_str)
        
def generate_multi_task_data():
    data_loader = MultiTaskClassifyDataPrepare(multi_task_config)  
    data_dir = multi_task_config.get("data_dir")
    for file_str in ['train', 'dev', 'test']:
        data_file = os.path.join(data_dir, multi_task_config.get("{}_data".format(file_str)))
        data_loader.gen_npy_data(data_file, file_str)