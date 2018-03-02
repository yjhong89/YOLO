import tensorflow as tf
import argparse
import configparser
from train import train
from detection import detect
from make_tfrecord import load_config
import os
import pandas as pd

# Not int, must be string
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=['config.ini'])
    parser.add_argument('-t', '--data_type', nargs='+', default=['train','val'])
    parser.add_argument('-d', '--delete', action='store_true')
    parser.add_argument('--train', type=str2bool, default='t')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_norm', type=float, default=5.0)
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--save_secs', type=int, default=1000)
    parser.add_argument('--summary_secs', type=int, default=100)
    parser.add_argument('--logging_level', default='INFO')
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--probability_threshold', type=float, default=0.3)
    parser.add_argument('--iou_threshold', type=float, default=0.4)

    args = parser.parse_args()
    if args.logging_level:
        tf.logging.set_verbosity(args.logging_level)
    config = configparser.ConfigParser()
    load_config(config, args.config)

    model_name = config.get('config', 'model')

    base_dir = os.path.expanduser(config.get('config', 'basedir'))
    anchor_info = pd.read_csv(os.path.join(base_dir, config.get('cache', 'anchor'))).values

    if args.train:
        tf.logging.info('Training')
        train(config, args, anchor_info, model_name)
    else:
        tf.logging.info('Object detecting')
        detect(config, args, anchor_info, model_name)

def str2bool(v):
    if v.lower() in ('t', 'true'):
        return True
    elif v.lower() in ('f', 'false'):
        return False
    else:
        raise ValueError('Not supported type')


if __name__ == "__main__":
    main()
