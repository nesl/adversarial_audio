"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import os, sys
import numpy as np
import sys
import tensorflow as tf
from speech_commands import label_wav

def load_graph(filename):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def load_labels(filename):
    return [line.rstrip() for line in tf.gfile.FastGFile(filename)]
        

def print_output(output_preds, labels):
    top_k = output_pred.argsort()[-5:][::-1]
    for node_id in top_k:
        human_string = labels[node_id]
        score = output_pred[node_id]
        print('%s %d score = %0.5f' %(human_string, node_id, score))
    print('----------------------')

################## GenAttack again ?
# TODO(malzantot): any thoughts about byte ordering ?
header_len = 44 + 1
mutation_p = 0.15
def gen_population_member(x_orig, bps=16):
    new_bytearray = bytearray(x_orig)
    step = 2
    if bps == 8:
        step = 1
    for i in range(header_len, len(x_orig), step):
        if np.random.random() < mutation_p:
            if np.random.random() < 0.5:
                new_bytearray[i] = min(255, new_bytearray[i]+1)
            else:
                new_bytearray[i] = max(0, new_bytearray[i]-1)
    return bytes(new_bytearray)

def crossover(x1, x2, bps=16):
    ba1 = bytearray(x1)
    ba2 = bytearray(x2)
    step = 2
    if bps == 8:
        step = 1
    for i in range(header_len, len(x1)):
        if np.random.random() < 0.5:
            ba2[i] = ba1[i]
    return bytes(ba2)

def refine(x_new, x_orig, pbs=16, limit=10):
    ba_new = bytearray(x_new)
    ba_orig = bytearray(x_orig)
    step = 2
    if pbs == 8:
        step = 1
    for i in range(header_len, len(x_new), step):
        # if np.random.random() < 0.5:
        ba_new[i] = min(ba_orig[i]+limit, max(ba_orig[i]-limit, ba_new[i]))
        ba_new[i] = min(255, max(0, ba_new[i]))
    return bytes(ba_new)

def mutation(x, pbs=16):
    ba = bytearray(x)
    step = 2
    if pbs == 8:
        step = 1
    for i in range(header_len, len(x), step):
        if np.random.random() < 0.05:
            ba[i] = max(0, min(255, np.random.choice(list(range(ba[i]-4, ba[i]+4)))))
        elif np.random.random() < 0.10:
            ba[i] = max(0, min(255, ba[i] + np.random.choice([-1, 1])))
    return bytes(ba)

def score(sess, x, target, input_tensor, output_tensor):
    output_preds, = sess.run(output_tensor,
        feed_dict={input_tensor: x})
    return output_preds[target]

def generate_attack(x_orig, target, limit, sess, input_node,
    output_node, max_iters, bps=16, verbose=False):
    pop_size = 10
    elite_size = 2
    refine_every = 5
    temp = 0.03
    initial_pop = [gen_population_member(x_orig, bps) for _ in range(pop_size)]
    for idx in range(max_iters):
        if idx > 0 and idx % refine_every == 0:
            initial_pop = [refine(x, x_orig, bps, limit=limit) for x in initial_pop]
        pop_scores = np.array([score(sess, x, target, input_node, output_node) for x in initial_pop])
        pop_ranks = list(reversed(np.argsort(pop_scores)))
        elite_set = [initial_pop[x] for x in pop_ranks[:elite_size]]
        top_attack = initial_pop[pop_ranks[0]]
        if verbose:
            print("%03d -- %0.2f" %(idx+1, np.max(pop_scores)))
        scores_logits = np.exp(pop_scores/temp)
        pop_probs = scores_logits / np.sum(scores_logits)
        child_set = [crossover(
            initial_pop[np.random.choice(pop_size, p=pop_probs)],
            initial_pop[np.random.choice(pop_size, p=pop_probs)], bps)
            for _ in range(pop_size - elite_size)]
        initial_pop = elite_set + [mutation(child, bps) for child in child_set]
    return top_attack
        
def save_audiofile(output, filename):        
    with open(filename, 'wb') as fh:
        fh.write(output)

def load_audiofile(filename):
    with open(filename, 'rb') as fh:
        return fh.read()

flags = tf.flags
flags.DEFINE_string("data_dir", "", "Data dir")
flags.DEFINE_string("output_dir", "", "Data dir")
flags.DEFINE_string("target_label", "", "Target classification label")
flags.DEFINE_integer("limit", 4, "Noise limit")
flags.DEFINE_string("graph_path", "", "Path to frozen graph file.")
flags.DEFINE_string("labels_path", "", "Path to labels file.")
flags.DEFINE_boolean("verbose", False, "")
flags.DEFINE_integer("max_iters", 200, "Maxmimum number of iterations")
FLAGS = flags.FLAGS

if __name__ == '__main__':
    data_dir = FLAGS.data_dir
    output_dir = FLAGS.output_dir
    target_label = FLAGS.target_label
    eps_limit = FLAGS.limit
    graph_path = FLAGS.graph_path
    labels_path = FLAGS.labels_path
    max_iters = FLAGS.max_iters
    verbose = FLAGS.verbose
    input_node_name = 'wav_data:0'
    output_node_name = 'labels_softmax:0'

    labels = load_labels(labels_path)

    wav_files_list =\
        [f for f in os.listdir(data_dir) if f.endswith(".wav")]
    
    target_idx = [idx for idx in range(len(labels)) if labels[idx]==target_label]
    if len(target_idx) == 0:
        print("Target label not found.")
        sys.exit(1)
    target_idx = target_idx[0]

    load_graph(graph_path)
    with tf.Session() as sess:
        output_node = sess.graph.get_tensor_by_name(output_node_name) 
        for input_file in wav_files_list:
            x_orig = load_audiofile(data_dir+'/'+input_file)
            #TODO(malzantot) fix
            # x_pbs = 1
            x_pbs=  int(x_orig[34])
            num_channels = int(x_orig[22])
            attack_output = generate_attack(x_orig, target_idx, eps_limit,
                sess, input_node_name, output_node, max_iters, x_pbs, verbose)
            save_audiofile(attack_output, output_dir+'/'+input_file)
                
       



