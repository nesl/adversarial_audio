"""
Author: Moustafa Alzantot (malzantot@ucla.edu)

"""

import numpy as np
import tensorflow as tf
from speech_commands import label_wav
import os, sys
import csv
flags = tf.flags
flags.DEFINE_string('output_dir', '', 'output data directory')
flags.DEFINE_string('labels_file', '', 'Labels file.')
flags.DEFINE_string('graph_file', '', '')
flags.DEFINE_string('output_file', 'eval_output.csv', 'CSV file of evaluation results')
FLAGS = flags.FLAGS

def load_graph(filename):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def load_labels(filename):
    return [line.rstrip() for line in tf.gfile.FastGFile(filename)]


def load_audiofile(filename):
    with open(filename, 'rb') as fh:
        return fh.read()

if __name__ == '__main__':
    output_dir = FLAGS.output_dir
    labels_file = FLAGS.labels_file
    graph_file = FLAGS.graph_file
    output_file = FLAGS.output_file
    labels = load_labels(labels_file)
    n_labels = len(labels)
    result_mat = np.zeros((n_labels, n_labels))
    input_node_name = 'wav_data:0'
    output_node_name = 'labels_softmax:0'
    load_graph(graph_file)
    
    ## Header of output file
    output_fh = open(output_file, 'w')
    fieldnames = ['filename', 'original', 'target', 'predicted']
    for label in labels:
        fieldnames.append(label)
    csv_writer = csv.DictWriter(output_fh, fieldnames=fieldnames)
    print(fieldnames)
    csv_writer.writeheader()
    with tf.Session() as sess:
        output_node = sess.graph.get_tensor_by_name(output_node_name) 
        for src_idx, src_label in enumerate(labels):
            for target_idx, target_label in enumerate(labels):
                case_dir = format("%s/%s/%s" %(output_dir, target_label, src_label))
                if os.path.exists(case_dir):
                    wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
                    for wav_filename in wav_files:
                        wav_data = load_audiofile(wav_filename)
                        preds = sess.run(output_node, feed_dict = {
                                input_node_name: wav_data
                        })
                        wav_pred = np.argmax(preds[0])
                        if wav_pred == target_idx:
                            result_mat[src_idx][wav_pred] += 1
                        row_dict = dict()
                        row_dict['filename'] = wav_filename
                        row_dict['original'] = src_label
                        row_dict['target'] = target_label
                        row_dict['predicted'] = labels[wav_pred]
                        for i in range(preds[0].shape[0]):
                            row_dict[labels[i]] = preds[0][i]
                        csv_writer.writerow(row_dict)
        
        print(result_mat)
        print(np.sum(result_mat))
                        

