"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
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

header_len = 44
mutation_p = 0.05
def gen_population_member(x_orig):
    new_bytearray = bytearray(x_orig)
    for i in range(header_len, len(x_orig)):
        if np.random.random() < mutation_p:
            if np.random.random() < 0.5:
                new_bytearray[i] = min(255, new_bytearray[i]+1)
            else:
                new_bytearray[i] = max(0, new_bytearray[i]-1)
    return bytes(new_bytearray)

def crossover(x1, x2):
    ba1 = bytearray(x1)
    ba2 = bytearray(x2)
    for i in range(header_len, len(x1)):
        if np.random.random() < 0.5:
            ba2[i] = ba1[i]
    return bytes(ba2)

def mutation(x):
    ba = bytearray(x)
    for i in range(header_len, len(x)):
        if np.random.random() < 0.05:
            ba[i] = max(0, min(255, np.random.choice(list(range(ba[i]-4, ba[i]+4)))))
        elif np.random.random() < 0.10:
            ba[i] = max(0, min(255, ba[i] + np.random.choice([-1, 1])))
    return bytes(ba)

def score(sess, x, target, input_tensor, output_tensor):
    output_preds, = sess.run(output_tensor,
        feed_dict={input_tensor: x})
    return output_preds[target]

if __name__ == '__main__':
    wav_filename = 'speech_dataset/yes/ce49cb60_nohash_1.wav'
    output_dir = 'output/'
    graph_filename = 'frozen_graph/my_frozen_graph.pb'
    labels_filename = 'ckpts/conv_labels.txt'
    input_node_name = 'wav_data:0'
    output_node_name = 'labels_softmax:0'

    # label_wav.label_wav(wav_filename, labels_filename, 
    #    graph_filename, input_node_name, output_node_name, 7)
    load_graph(graph_filename)
    # write graph summary
    # writer = tf.summary.FileWriter('my_log', tf.get_default_graph())
    # writer.flush()
    # writer.close()

    labels = load_labels(labels_filename)
    with open(wav_filename, 'rb') as wav_file:
        wav_data = wav_file.read()
        
    with open(output_dir + 'attack_original.wav', 'wb') as fh:
        fh.write(wav_data)
    
    with tf.Session() as sess:
        # mfcc_tensor = sess.graph.get_tensor_by_name('Mfcc:0')
        output_tensor = sess.graph.get_tensor_by_name(output_node_name) 
        output_pred, = sess.run(output_tensor, feed_dict = {
            input_node_name: wav_data})
        print_output(wav_data, labels)

        pop_size = 20
        elite_size = 2
        target = 3 # 3 is no
        max_iter = 200
        refine_every = 10
        initial_pop = [gen_population_member(wav_data) for _ in range(pop_size)]
        print('Original target score = ' , score(sess, wav_data, target, input_node_name, output_tensor))
        # print('Original target score = ' , score(sess, wav_data, target, input_node_name, output_tensor))
        for idx in range(max_iter):
            if idx % refine_every == 0:
                print ("Refining population")
                # initial_pop = [crossover(wav_data, x) for x in initial_pop]
            pop_scores = np.array([score(sess, x, target, input_node_name, output_tensor) for x in initial_pop])
            print(idx, np.max(pop_scores))            
            pop_ranks = list(reversed(np.argsort(pop_scores)))
            elite_set = [initial_pop[x] for x in pop_ranks[:elite_size]]
            top_attack = initial_pop[pop_ranks[0]]
            temp = 0.01
            scores_logits = np.exp(pop_scores/temp)
            pop_probs = scores_logits / np.sum(scores_logits)
            print('\r\t\t\t', idx, '    (', np.min(pop_probs), ' : ', np.max(pop_probs), ') ')
            print(np.sum(pop_probs))

            child_set = [crossover(
                initial_pop[np.random.choice(pop_size, p=pop_probs)],
                initial_pop[np.random.choice(pop_size, p=pop_probs)],

            ) for _ in range(pop_size - elite_size)]
            initial_pop = elite_set + [mutation(child) for child in child_set]
            if idx % 50 == 0:
                print('Orig score = %0.5f  - Target score = %0.5f' 
                %(
                score(sess, top_attack, 2, input_node_name, output_tensor),
                score(sess, top_attack, target, input_node_name, output_tensor)

                )
                )
                with open(output_dir + 'attack_%03d.wav' %idx, 'wb') as fh:
                    fh.write(top_attack)

        with open(output_dir + 'attack_final.wav', 'wb') as fh:
            fh.write(top_attack)
       



