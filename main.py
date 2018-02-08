import tensorflow as tf
import argparse
import configparser
from train import train
from make_tfrecord import load_config
import os

# Not int, must be string
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=['config.ini'])
    parser.add_argument('-t', '--data_type', nargs='+', default=['train','val'])
    parser.add_argument('--steps', type=int, default=100000000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_norm', type=float, default=5.0)
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--save_secs', type=int, default=1000)
    parser.add_argument('--summary_secs', type=int, default=100)
    parser.add_argument('--logging_level', default='INFO')
    parser.add_argument('--task', type=int, default=0)

    args = parser.parse_args()
    if args.logging_level:
        tf.logging.set_verbosity(args.logging_level)
    config = configparser.ConfigParser()
    load_config(config, args.config)

    #print(config)
    
    train(config, args)

if __name__ == "__main__":
    main()
