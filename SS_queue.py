#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse, json, unicodedata
import pdb
import time
import multiprocessing as mp
from threading import Thread
import itertools
#from six.moves import range, zip
from six.moves.queue import Queue
import uuid

#from tensorpack import *
#from tensorpack.tfutils.symbolic_functions import *
#from tensorpack.tfutils.summary import *

#from tensorflow.models.rnn import rnn_cell
rnn_cell = tf.nn.rnn_cell
rnn = tf.nn.rnn
from keras.preprocessing import sequence
from cocoeval import COCOScorer
from sklearn.metrics import average_precision_score
from collections import defaultdict
gpu_id = 0

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Extract a CNN features')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--net', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset to extract',
                        default='train_val', type=str)
    parser.add_argument('--task', dest='task',
                        help='train or test',
                        default='train', type=str)
    parser.add_argument('--tg', dest='tg',
                        help='target to be extract lstm feature',
                        default='/home/Hao/tik/jukin/data/h5py', type=str)
    parser.add_argument('--ft', dest='ft',
                        help='choose which feature type would be extract',
                        default='lstm1', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

class Video_Caption_Generator():
    def __init__(self, dim_image, dim_tracker, n_words, dim_hidden, batch_size, n_lstm_steps, tracker_cnt, drop_out_rate, bias_init_vector=None):
	self.dim_image = dim_image
	self.dim_tracker = dim_tracker
	self.n_words = n_words
	self.dim_hidden = dim_hidden
	self.batch_size = batch_size
	self.n_lstm_steps = n_lstm_steps
	self.tracker_cnt = tracker_cnt
	self.drop_out_rate = drop_out_rate

	with tf.device("/cpu:0"):
	    self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')
	#self.Wemb_W = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb_W')
	#self.Wemb_b = tf.Variable(tf.random_uniform([dim_hidden], -0.1, 0.1), name='Wemb_b')

	#self.lstm1 = rnn_cell.BasicLSTMCell(dim_hidden)
	self.lstm1 = rnn_cell.LSTMCell(self.dim_hidden,2*self.dim_hidden,use_peepholes = True, initializer=tf.random_uniform_initializer(-0.1,0.1))
	self.lstm1_dropout = rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.drop_out_rate)
	#self.lstm2 = rnn_cell.BasicLSTMCell(dim_hidden)
	self.lstm2 = rnn_cell.LSTMCell(self.dim_hidden,2*self.dim_hidden,use_peepholes = True, initializer=tf.random_uniform_initializer(-0.1,0.1))
	self.lstm2_dropout = rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)

	self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
	self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')
	self.encode_tracker_W = tf.Variable( tf.random_uniform([dim_tracker, dim_hidden], -0.1, 0.1), name='encode_tracker_W')
	self.encode_tracker_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_tracker_b')
	self.embed_att_w = tf.Variable(tf.random_uniform([dim_hidden, 1], -0.1,0.1), name='embed_att_w')
        self.embed_att_Wa = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1,0.1), name='embed_att_Wa')
	self.embed_att_Ua = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden],-0.1,0.1), name='embed_att_Ua')
	self.embed_att_ba = tf.Variable( tf.zeros([dim_hidden]), name='embed_att_ba')

	self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
	if bias_init_vector is not None:
	    self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
	else:
	    self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
	video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image])
	video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

	tracker = tf.placeholder(tf.float32, [self.batch_size, self.tracker_cnt, self.dim_tracker])
	tracker_mask = tf.placeholder(tf.float32, [self.batch_size, self.tracker_cnt])

	caption = tf.placeholder(tf.int32, [self.batch_size, 35])
	caption_mask = tf.placeholder(tf.float32, [self.batch_size, 35])

	video_flat = tf.reshape(video, [-1, self.dim_image])
	image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (batch_size*n_lstm_steps, dim_hidden)
	image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

	tracker_flat = tf.reshape(tracker, [-1, self.dim_tracker])
	tracker_emb = tf.nn.xw_plus_b( tracker_flat, self.encode_tracker_W, self.encode_tracker_b) # (batch_size*n_lstm_steps, dim_hidden)
	tracker_emb = tf.reshape(tracker_emb, [self.batch_size, self.tracker_cnt, self.dim_hidden])
	tracker_emb = tf.transpose(tracker_emb, [1,0,2])

	state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
	state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
	state1_return = tf.zeros([self.batch_size, self.lstm1.state_size])
	state2_return = tf.zeros([self.batch_size, self.lstm2.state_size])
	padding = tf.zeros([self.batch_size, self.dim_hidden])

	loss_caption = 0.0
	state1_temp=[]
	state2_temp=[]
	for i in range(self.n_lstm_steps): ## Phase 1 => only read frames
	    if i > 0:
		tf.get_variable_scope().reuse_variables()

	    with tf.variable_scope("LSTM1"):
		output1, state1 = self.lstm1_dropout( tf.concat(1,[padding, image_emb[:,i,:]]), state1 )

	    with tf.variable_scope("LSTM2"):
		output2, state2 = self.lstm2_dropout( tf.concat(1,[padding, output1]), state2 )

	    state1_temp.append(state1)
	    state2_temp.append(state2)

	state1_temp=tf.pack(state1_temp)
	mask_local = tf.to_float(video_mask)
	mask_local = tf.expand_dims(mask_local,2)
	mask_local = tf.tile(mask_local,tf.constant([1,1,self.lstm1.state_size]))
	mask_local = tf.transpose(mask_local, perm=[1, 0, 2])
	state1 = tf.reduce_sum(tf.mul(state1_temp, mask_local),0)
	state2_temp=tf.pack(state2_temp)
	mask_local = tf.to_float(video_mask)
	mask_local = tf.expand_dims(mask_local,2)
	mask_local = tf.tile(mask_local,tf.constant([1,1,self.lstm2.state_size]))
	mask_local = tf.transpose(mask_local, perm=[1, 0, 2])
	state2 = tf.reduce_sum(tf.mul(state2_temp, mask_local),0)
	h_prev = tf.zeros([self.batch_size, self.dim_hidden])

	current_embed = tf.zeros([self.batch_size, self.dim_hidden])
	brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.tracker_cnt,1,1]) # n x h x 1
	image_part = tf.batch_matmul(tracker_emb, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.tracker_cnt,1,1])) + self.embed_att_ba # n x b x h
	for j in range(35): ## Phase 2 => only generate captions
	    e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part) # n x b x h
	    e = tf.batch_matmul(e, brcst_w)
	    e = tf.reduce_sum(e,2) # n x b
	    e_hat_exp = tf.mul(tf.transpose(tracker_mask), tf.exp(e)) # n x b
	    denomin = tf.reduce_sum(e_hat_exp,0) # b
	    denomin = denomin + tf.to_float(tf.equal(denomin, 0))
	    alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h
	    attention_list = tf.mul(alphas, tracker_emb) # n x b x h
	    atten = tf.reduce_sum(attention_list,0) # b x h

	    tf.get_variable_scope().reuse_variables()
	    with tf.variable_scope("LSTM1"):
		output1, state1 = self.lstm1_dropout( tf.concat(1,[atten,padding]), state1 )

	    with tf.variable_scope("LSTM2"):
		output2, state2 = self.lstm2_dropout( tf.concat(1,[current_embed, output1]), state2 )
	    h_prev = output2

	    labels = tf.expand_dims(caption[:,j], 1)
	    indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
	    concated = tf.concat(1, [indices, labels])
	    with tf.device('/cpu:0'):
		onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)
	    #current_embed = tf.matmul(onehot_labels,self.Wemb_W) + self.Wemb_b # b x h
	    with tf.device("/cpu:0"):
		current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,j])

	    logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
	    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
	    cross_entropy = cross_entropy * caption_mask[:,j]

	    current_loss = tf.reduce_sum(cross_entropy)
	    loss_caption += current_loss

	loss_caption = loss_caption / tf.reduce_sum(caption_mask)
	loss = loss_caption
        return loss, video, video_mask, tracker, tracker_mask, caption, caption_mask


    def build_generator(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

	tracker = tf.placeholder(tf.float32, [self.batch_size, self.tracker_cnt, self.dim_tracker])
	tracker_mask = tf.placeholder(tf.float32, [self.batch_size, self.tracker_cnt])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

	tracker_flat = tf.reshape(tracker, [-1, self.dim_tracker])
	tracker_emb = tf.nn.xw_plus_b( tracker_flat, self.encode_tracker_W, self.encode_tracker_b) # (batch_size*n_lstm_steps, dim_hidden)
	tracker_emb = tf.reshape(tracker_emb, [self.batch_size, self.tracker_cnt, self.dim_hidden])
	tracker_emb = tf.transpose(tracker_emb, [1,0,2])

	state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
	state1_return = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2_return = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        loss_caption = 0.0
	state1_temp=[]
	state2_temp=[]
        for i in range(self.n_lstm_steps): ## Phase 1 => only read frames
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
		output1, state1 = self.lstm1_dropout( tf.concat(1,[padding, image_emb[:,i,:]]), state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout( tf.concat(1,[padding, output1]), state2 )

	    state1_temp.append(state1)
	    state2_temp.append(state2)

	state1_temp=tf.pack(state1_temp)
	mask_local = tf.to_float(video_mask)
	mask_local = tf.expand_dims(mask_local,2)
	mask_local = tf.tile(mask_local,tf.constant([1,1,self.lstm1.state_size]))
	mask_local = tf.transpose(mask_local, perm=[1, 0, 2])
	state1 = tf.reduce_sum(tf.mul(state1_temp, mask_local),0)
	state2_temp=tf.pack(state2_temp)
	mask_local = tf.to_float(video_mask)
	mask_local = tf.expand_dims(mask_local,2)
	mask_local = tf.tile(mask_local,tf.constant([1,1,self.lstm2.state_size]))
	mask_local = tf.transpose(mask_local, perm=[1, 0, 2])
	state2 = tf.reduce_sum(tf.mul(state2_temp, mask_local),0)
	h_prev = tf.zeros([self.batch_size, self.dim_hidden])

	generated_words = []
	current_embed = tf.zeros([self.batch_size, self.dim_hidden])
	brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.tracker_cnt,1,1]) # n x h x 1
	image_part = tf.batch_matmul(tracker_emb, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.tracker_cnt,1,1])) + self.embed_att_ba # n x b x h
	for j in range(35): ## Phase 2 => only generate captions
	    e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part) # n x b x h
	    e = tf.batch_matmul(e, brcst_w)
	    e = tf.reduce_sum(e,2) # n x b
	    e_hat_exp = tf.mul(tf.transpose(tracker_mask), tf.exp(e)) # n x b
	    denomin = tf.reduce_sum(e_hat_exp,0) # b
	    denomin = denomin + tf.to_float(tf.equal(denomin, 0))
	    alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h
	    attention_list = tf.mul(alphas, tracker_emb) # n x b x h
	    atten = tf.reduce_sum(attention_list,0) # b x h

	    tf.get_variable_scope().reuse_variables()
	    with tf.variable_scope("LSTM1") as vs:
		output1, state1 = self.lstm1_dropout( tf.concat(1,[atten,padding]), state1 )
		lstm1_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

	    with tf.variable_scope("LSTM2") as vs:
		output2, state2 = self.lstm2_dropout( tf.concat(1,[current_embed, output1]), state2 )
		lstm2_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
	    h_prev = output2

	    logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b) # b x w
            max_prob_index = tf.argmax(logit_words, 1) # b
            generated_words.append(max_prob_index) # b

	    #current_embed = tf.matmul(logit_words,self.Wemb_W) + self.Wemb_b # b x h
	    with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)

	with tf.device("/cpu:0"):
		generated_words = tf.transpose(tf.pack(generated_words))
        return video, video_mask, tracker, tracker_mask, generated_words, lstm1_variables, lstm2_variables


############### Global Parameters ###############
video_data_path_train = '/home/PaulChen/MSRVTT/hdf5_generator/msrvtt_data/data_ch10/h5py_res+c3d/cont_captions/train.txt'
video_data_path_val = '/home/PaulChen/MSRVTT/hdf5_generator/msrvtt_data/data_ch10/h5py_res+c3d/cont_captions/val.txt'
video_data_path_test = '/home/PaulChen/MSRVTT/hdf5_generator/msrvtt_data/data_ch10/h5py_res+c3d/cont_captions/test.txt'
video_feat_path = '/home/PaulChen/MSRVTT/hdf5_generator/msrvtt_data/data_ch10/h5py_res+c3d/cont_captions/'
model_path = '/home/PaulChen/MSRVTT/tf_models/SS_Res+c3d/models_2'

############## Train Parameters #################
dim_image = 6144#4096*2
dim_tracker = 4096+4
dim_hidden= 500
n_frame_step = 45
tracker_cnt = 10
n_epochs = 200
batch_size = 100
learning_rate = 0.0001 #0.001
##################################################

def get_video_data(video_data_path, video_feat_path, train_ratio=0.9):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = video_data['video_path'].unique()
    train_len = int(len(unique_filenames)*train_ratio)

    train_vids = unique_filenames[:train_len]
    test_vids = unique_filenames[train_len:]

    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data

def get_video_data_HL(video_data_path, video_feat_path):
    files = open(video_data_path)
    List = []
    for ele in files:
        List.append(ele[:-1])
    return np.array(List)

def get_video_data_jukin(video_data_path_train, video_data_path_val, video_data_path_test):
    video_list_train = get_video_data_HL(video_data_path_train, video_feat_path)
    title = []
    fname = []
    for ele in video_list_train:
        batch_data = h5py.File(ele)
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in xrange(len(batch_fname)):
                fname.append(batch_fname[i])
                title.append(batch_title[i])
        batch_data.close()

    video_list_val = get_video_data_HL(video_data_path_val, video_feat_path)
    for ele in video_list_val:
        batch_data = h5py.File(ele)
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in xrange(len(batch_fname)):
                fname.append(batch_fname[i])
                title.append(batch_title[i])
        batch_data.close()

    '''
    video_list_test = get_video_data_HL(video_data_path_test, video_feat_path)
    for ele in video_list_test:
        batch_data = h5py.File(ele)
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in xrange(len(batch_fname)):
                fname.append(batch_fname[i])
                title.append(batch_title[i])
    '''
    video_list_test = []

    fname = np.array(fname)
    title = np.array(title)
    video_data = pd.DataFrame({'Description':title})

    return video_data, video_list_train, video_list_val, video_list_test

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector


def preProBuildLabel():
    ixtoword = {}
    wordtoix = {}
    ix = 1
    for w in range(1):
	wordtoix[w] = ix
	ixtoword[ix] = w
	ix += 1
    return wordtoix, ixtoword

def testing_one_multi_gt(sess, video_feat_path, ixtoword, counter, video_tf, video_mask_tf, tracker_tf, tracker_mask_tf, caption_tf):
    pred_sent = []
    gt_sent = []
    IDs = []
    namelist = []
    #print video_feat_path
    test_data_batch = h5py.File(video_feat_path)
    gt_captions = json.load(open('msrvtt_vid2caption_clean.json'))  #test_data_batch['title'] 
    
    video_feat = np.zeros((batch_size, n_frame_step, dim_image))
    video_mask = np.zeros((batch_size, n_frame_step))
    tracker = np.zeros((batch_size, tracker_cnt, dim_tracker))
    tracker_mask = np.zeros((batch_size, tracker_cnt))
    if 'tracker' in test_data_batch.keys():
        tracker = np.array(test_data_batch['tracker'])
    if 'tracker_mask' in test_data_batch.keys():
        tracker_mask = np.array(test_data_batch['tracker_mask'])
   
    for ind in xrange(batch_size):
        video_feat[ind,:,:] = test_data_batch['data'][:n_frame_step,ind,:]
        idx = np.where(test_data_batch['label'][:,ind] != -1)[0]
        if(len(idx) == 0):
                continue
        video_mask[ind,idx[-1]] = 1.
    generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask,tracker_tf:tracker, tracker_mask_tf:tracker_mask})
    
    #print video_feat_path
    for ind in xrange(batch_size):
        cap_key = test_data_batch['fname'][ind]
        if cap_key == '':
            continue
        else:
            generated_words = ixtoword[generated_word_index[ind]]
            punctuation = np.argmax(np.array(generated_words) == '.')+1
            generated_words = generated_words[:punctuation]
            generated_sentence = ' '.join(generated_words)
            pred_sent.append([{'image_id':str(counter),'caption':generated_sentence}])
            namelist.append(cap_key)
            for i,s in enumerate(gt_captions[cap_key]):
                s = unicodedata.normalize('NFKD', s).encode('ascii','ignore')
                gt_sent.append([{'image_id':str(counter),'cap_id':i,'caption':s}])
                IDs.append(str(counter))

            counter += 1
    return pred_sent, gt_sent, IDs, counter, namelist

def testing_all_multi_gt(sess, test_data, ixtoword, video_tf, video_mask_tf, tracker_tf, tracker_mask_tf, caption_tf):
    pred_sent = []
    gt_sent = []
    IDs_list = []
    flist = []
    counter = 0
    gt_dict = defaultdict(list)
    pred_dict = {}
    for _, video_feat_path in enumerate(test_data):
            [b,c,d,counter,fns] = testing_one_multi_gt(sess, video_feat_path, ixtoword, counter, video_tf, video_mask_tf, tracker_tf, tracker_mask_tf, caption_tf)
            pred_sent += b
            gt_sent += c
            IDs_list += d
            flist += fns

    for k,v in zip(IDs_list,gt_sent):
        gt_dict[k].append(v[0])

    new_flist = []
    new_IDs_list = []
    for k,v in zip(range(len(pred_sent)),pred_sent):
#        print str(k)+': '+str(v)
        if flist[k] not in new_flist:
            new_flist.append(flist[k])
            new_IDs_list.append(str(k))
            pred_dict[str(k)] = v


    return pred_sent, gt_sent, new_IDs_list, gt_dict, pred_dict, flist

def load_data_into_queue(train_data, data_queue, hdf5_key):
    current_batch_file_idx = 0
    while current_batch_file_idx < len(train_data):
        if data_queue.full()==False:
            current_batch = h5py.File(train_data[current_batch_file_idx])
            if hdf5_key not in current_batch.keys():
                if hdf5_key == 'data':
                    data_queue.put(np.zeros((n_frame_step, batch_size, dim_image)))
                elif hdf5_key == 'tracker':
                    data_queue.put(np.zeros((batch_size, tracker_cnt, dim_tracker)))
            else:
                data_queue.put(np.array(current_batch[hdf5_key]))

            current_batch_file_idx += 1
            current_batch.close()
        
def train():
    meta_data, train_data, val_data, test_data = get_video_data_jukin(video_data_path_train, video_data_path_val, video_data_path_test)
    captions = meta_data['Description'].values
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=1)

#    np.save('./data'+str(gpu_id)+'/ixtoword', ixtoword)
#    np.save('./data'+str(gpu_id)+'/wordtoix', wordtoix)
#    sys.exit()
    ixtoword=pd.Series(np.load('./data_all/ixtoword.npy').tolist())
    wordtoix=pd.Series(np.load('./data_all/wordtoix.npy').tolist())

    model = Video_Caption_Generator(
            dim_image=dim_image,
	    dim_tracker=dim_tracker,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
	    tracker_cnt=tracker_cnt,
            drop_out_rate = 0.5,
            bias_init_vector=None)

    tf_loss, tf_video, tf_video_mask, tf_tracker, tf_tracker_mask, tf_caption, tf_caption_mask= model.build_model()
    #loss_summary = tf.scalar_summary("Loss",tf_loss)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    #merged = tf.merge_all_summaries()
    #writer = tf.train.SummaryWriter('/tmp/tf_log', sess.graph_def)

    with tf.device("/cpu:0"):
    	saver = tf.train.Saver(max_to_keep=100)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.initialize_all_variables().run()
    saver.restore(sess, 'models/model-0')

    tStart_total = time.time()
    nr_prefetch = int(3)
    for epoch in range(n_epochs):
        index = np.arange(len(train_data))
        np.random.shuffle(index)
        train_data = train_data[index]
        
        tStart_epoch = time.time()
        loss_epoch = np.zeros(len(train_data))
        ## init queue
        data_queue = mp.Queue(nr_prefetch)
#        tracker_queue = mp.Queue(nr_prefetch)
        title_queue = mp.Queue(nr_prefetch)
        t1 = Thread(target=load_data_into_queue, args=(train_data, data_queue, 'data'))
#        t2 = Thread(target=load_data_into_queue, args=(train_data, tracker_queue, 'tracker'))
        t3 = Thread(target=load_data_into_queue, args=(train_data, title_queue, 'title'))
        t1.start()
#        t2.start()
        t3.start()
        for current_batch_file_idx in range(len(train_data)):
            tStart = time.time()
            current_batch = h5py.File(train_data[current_batch_file_idx])
            current_feats = np.zeros((batch_size, n_frame_step, dim_image))
            current_video_masks = np.zeros((batch_size, n_frame_step))
            current_video_len = np.zeros(batch_size)
            
            if 'tracker' in current_batch.keys():
                current_tracker = np.array(current_batch['tracker'])
            else:
                current_tracker = np.zeros((batch_size, tracker_cnt, dim_tracker))
            
            if 'tracker_mask' in current_batch.keys():
                current_tracker_mask = np.array(current_batch['tracker_mask'])
            else:
                current_tracker_mask = np.zeros((batch_size, tracker_cnt))

#            current_tracker = tracker_queue.get()
            current_batch_data = data_queue.get()
            current_batch_title = title_queue.get()
            for ind in xrange(batch_size):
                current_feats[ind,:,:] = current_batch_data[:,ind,:]
                idx = np.where(current_batch['label'][:,ind] != -1)[0]
                if len(idx) == 0:
                        continue
                current_video_masks[ind,idx[-1]] = 1

            current_captions = current_batch_title
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=35-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            current_batch.close()


            _, loss_val= sess.run(
                [train_op, tf_loss],
                feed_dict={
                tf_video: current_feats,
                tf_video_mask : current_video_masks,
                tf_tracker : current_tracker,
                tf_tracker_mask : current_tracker_mask,
                tf_caption: current_caption_matrix,
                tf_caption_mask: current_caption_masks
                })
            #writer.add_summary(summary_str, epoch)
            loss_epoch[current_batch_file_idx] = loss_val
            tStop = time.time()
            #print "Epoch:", epoch, " Batch:", current_batch_file_idx, " Loss:", loss_val
            #print "Time Cost:", round(tStop - tStart,2), "s"

        t1.join()
#       t2.join()
        t3.join()
        print "Epoch:", epoch, " done. Loss:", np.mean(loss_epoch)
        tStop_epoch = time.time()
        print "Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s"
	sys.stdout.flush()

        if np.mod(epoch, 2) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
    	    with tf.device('/cpu:0'):
            	saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
        if np.mod(epoch, 10) == 0:
            current_batch = h5py.File(val_data[np.random.randint(0,len(val_data))])
            video_tf, video_mask_tf, tracker_tf, tracker_mask_tf, caption_tf, lstm1_variables_tf, lstm2_variables_tf = model.build_generator()
            ixtoword = pd.Series(np.load('./data_all/ixtoword.npy').tolist())
#            [pred_sent, gt_sent, id_list, gt_dict, pred_dict, fnamelist] = testing_all_multi_gt(sess, train_data[-2:], ixtoword,video_tf, video_mask_tf, tracker_tf, tracker_mask_tf, caption_tf)
#            for key in pred_dict.keys():
#                for ele in gt_dict[key]:
#                    print "GT:  " + ele['caption']
#                print "PD:  " + pred_dict[key][0]['caption']
#                print '-------'

            [pred_sent, gt_sent, id_list, gt_dict, pred_dict, fnamelist] = testing_all_multi_gt(sess, val_data, ixtoword,video_tf, video_mask_tf, tracker_tf, tracker_mask_tf, caption_tf)

            scorer = COCOScorer()
            total_score = scorer.score(gt_dict, pred_dict, id_list)

    print "Finally, saving the model ..."
    with tf.device('/cpu:0'):
	    saver.save(sess, os.path.join(model_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print "Total Time Cost:", round(tStop_total - tStart_total,2), "s"

def test(model_path='models/model-900', video_feat_path=video_feat_path):
    meta_data, train_data, val_data, test_data = get_video_data_jukin(video_data_path_train, video_data_path_val, video_data_path_test)
    test_data = val_data
    ixtoword = pd.Series(np.load('./data_all/ixtoword.npy').tolist())

    model = Video_Caption_Generator(
            dim_image=dim_image,
	    dim_tracker=dim_tracker,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
	    tracker_cnt=tracker_cnt,
            drop_out_rate = 0,
            bias_init_vector=None)

    video_tf, video_mask_tf, tracker_tf, tracker_mask_tf, caption_tf, lstm1_variables_tf, lstm2_variables_tf = model.build_generator()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    for ind, row in enumerate(lstm1_variables_tf):
        if ind % 4 == 0:
                assign_op = row.assign(tf.mul(row,1-0.5))
                sess.run(assign_op)
    for ind, row in enumerate(lstm2_variables_tf):
        if ind % 4 == 0:
                assign_op = row.assign(tf.mul(row,1-0.5))
                sess.run(assign_op)

#    [pred_sent, gt_sent] = testing_all(sess, test_data, ixtoword,video_tf, video_mask_tf, tracker_tf, tracker_mask_tf, caption_tf)
#    scorer = COCOScorer()
#    total_score = scorer.score(gt_sent, pred_sent, range(len(pred_sent)))
    [pred_sent, gt_sent, id_list, gt_dict, pred_dict, fnamelist] = testing_all_multi_gt(sess, test_data, ixtoword,video_tf, video_mask_tf, tracker_tf, tracker_mask_tf, caption_tf)
    np.savez('result/'+model_path.split('/')[1],gt = gt_sent,pred=pred_sent,fname=fnamelist)
    scorer = COCOScorer()
    total_score = scorer.score(gt_dict, pred_dict, id_list)
    return total_score

if __name__ == '__main__':
    args = parse_args()
    if args.task == 'train':
        with tf.device('/gpu:'+str(gpu_id)):
            train()
    elif args.task == 'test':
        with tf.device('/gpu:'+str(gpu_id)):
            total_score = test(model_path = args.model)
