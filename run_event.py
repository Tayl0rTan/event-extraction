import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # ERROR
import tensorflow as tf
from argparse import ArgumentParser
from train_helper import run_event_role_mrc, run_event_classification, run_multi_classification, run_multi_task_classification
from data_processing.gen_training_data import generate_type_data, generate_status_data, generate_role_data,generate_multi_task_data
tf.logging.set_verbosity(tf.logging.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_type", default="type", type=str)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--epochs", default=15, type=int)
    # bert lr
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--valid_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=4, type=int)
    parser.add_argument("--do_train", action='store_true', default=False)
    parser.add_argument("--do_test", action='store_true', default=False)
    parser.add_argument("--shuffle_buffer", default=128, type=int)
    parser.add_argument("--pre_buffer_size", default=16, type=int)
    parser.add_argument("--save_predict", type=str, default="save_predict")

    args = parser.parse_args()
    if args.model_type == "role":
        generate_role_data()
        run_event_role_mrc(args)
    elif args.model_type == "type":
        generate_type_data()
        run_event_classification(args)
    elif args.model_type == "status":
        generate_status_data()
        run_multi_classification(args)
    elif args.model_type == "multi_task":
        generate_multi_task_data()
        run_multi_task_classification(args)
        

if __name__ == '__main__':
    main()
