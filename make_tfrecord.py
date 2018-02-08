import os
import tensorflow as tf
import pandas as pd
import configparser
import argparse
import importlib
import shutil


def load_config(config, inis):
    # ini must be element of list
    for i in inis:
        # expanduser: change '~' to absolute value
        # expandvars: return the argument with environment variables expanded
        path = os.path.expanduser(os.path.expandvars(i))
        config.read(path)


def main():
    parser = argparse.ArgumentParser()
    # nargs: number of arguments, '+': more than '1'
    parser.add_argument('-c','--config', nargs='+', default=['config.ini'])
    # action: save boolean switch
    parser.add_argument('-v', '--verify', action='store_true')
    parser.add_argument('-t', '--data_type', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--log', default='INFO')
    parser.add_argument('-d','--delete', action='store_true')

    args = parser.parse_args()
    if args.log is 'INFO':
        # 'tf.logging.set_verbosity': print tf.logging.warn/info 
        tf.logging.set_verbosity(args.log)

    # configuring '.ini' file
    config = configparser.ConfigParser()
    # read config file
    load_config(config, args.config)   
   
    base_dir = os.path.expanduser(config.get('config', 'basedir')) 
    class_txt = os.path.join(base_dir, config.get('cache', 'name'))

    # Make directory where tfrecord files saved
    cache_dir = os.path.join(base_dir, config.get('cache', 'cachedir'))
    if args.delete:
        tf.logging.warn('Delete tfrecord files')
        shutil.rmtree(cache_dir)
    # exist_ok=True: Does not raise an exception if the directory already exists
    os.makedirs(cache_dir, exist_ok=True)

    with open(class_txt, 'r') as f:
        # .strip(): remove empty space in string, remove '\n'
        class_names = [line.strip() for line in f]
    class_names_idx = dict([(name, index) for index, name in enumerate(class_names)]) 

    data_path = os.path.join(base_dir, config.get('cache', 'dataset'))

    # os.path.splitext: abc.txt->['abc', 'txt']
    # pd.read_csv: read column value with index, voc.tsv-> root(header), row2,3 is information, row index is addes from 0
    datasets = (os.path.basename(os.path.splitext(data_path)[0]), pd.read_csv(data_path))
    print(datasets)

    # bluese05.tistory.com/31
    # dynamic import with module's name
    # importlib.import_module() = __import__()
    call_module = importlib.import_module('cache.read_data')
    
    for t in args.data_type:
        tfrecord_path = os.path.join(cache_dir, t+'.tfrecord')
        tf.logging.info('Writing %s.tfrecord' % (t))
        # tfrecord write module
        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            tf.logging.info('Loading %s %s dataset' % (datasets[0], t))
            # getattr(): Access to imported module's attribute
            function = getattr(call_module, datasets[0])
    
            # pd.read_csv.iterrows() returns (row_index, (key, value))        
            for i, row in datasets[1].iterrows():
                function(writer, class_names_idx, t, row, base_dir, args.verify)



if __name__ == "__main__":
    main()
    
    
