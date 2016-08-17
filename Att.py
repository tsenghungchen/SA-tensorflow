#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import pdb
import time
import json
from collections import defaultdict
from tensorflow.models.rnn import rnn, rnn_cell
from keras.preprocessing import sequence
from cocoeval import COCOScorer
import unicodedata
gpu_id = 5

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
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, drop_out_rate, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm3 = rnn_cell.LSTMCell(self.dim_hidden,2*self.dim_hidden,use_peepholes = True)
        self.lstm3_dropout = rnn_cell.DropoutWrapper(self.lstm3,output_keep_prob=1 - self.drop_out_rate)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')
        self.embed_att_w = tf.Variable(tf.random_uniform([dim_hidden, 1], -0.1,0.1), name='embed_att_w')
        self.embed_att_Wa = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1,0.1), name='embed_att_Wa')
        self.embed_att_Ua = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden],-0.1,0.1), name='embed_att_Ua')
        self.embed_att_ba = tf.Variable( tf.zeros([dim_hidden]), name='embed_att_ba')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        self.embed_nn_Wp = tf.Variable(tf.random_uniform([3*dim_hidden, dim_hidden], -0.1,0.1), name='embed_nn_Wp')
        self.embed_nn_bp = tf.Variable(tf.zeros([dim_hidden]), name='embed_nn_bp')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image]) # b x n x d
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps]) # b x n

        caption = tf.placeholder(tf.int32, [self.batch_size, n_caption_step]) # b x 16
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, n_caption_step]) # b x 16

        video_flat = tf.reshape(video, [-1, self.dim_image]) # (b x n) x d
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (b x n) x h
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden]) # b x n x h
        image_emb = tf.transpose(image_emb, [1,0,2]) # n x b x h

        state1 = tf.zeros([self.batch_size, self.lstm3.state_size]) # b x s
        h_prev = tf.zeros([self.batch_size, self.dim_hidden]) # b x h

        loss_caption = 0.0

        current_embed = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_lstm_steps,1,1]) # n x h x 1
        image_part = tf.batch_matmul(image_emb, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.n_lstm_steps,1,1])) + self.embed_att_ba # n x b x h
        for i in range(n_caption_step):
            e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part) # n x b x h
            e = tf.batch_matmul(e, brcst_w)    # unnormalized relevance score 
            e = tf.reduce_sum(e,2) # n x b
            e_hat_exp = tf.mul(tf.transpose(video_mask), tf.exp(e)) # n x b 
            denomin = tf.reduce_sum(e_hat_exp,0) # b
            denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
            alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h  # normalize to obtain alpha
            attention_list = tf.mul(alphas, image_emb) # n x b x h
            atten = tf.reduce_sum(attention_list,0) # b x h       #  soft-attention weighted sum
            if i > 0: tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM3"):
                output1, state1 = self.lstm3_dropout( tf.concat(1,[atten, current_embed]), state1 ) # b x h

            output2 = tf.tanh(tf.nn.xw_plus_b(tf.concat(1,[output1,atten,current_embed]), self.embed_nn_Wp, self.embed_nn_bp)) # b x h
            h_prev = output1 # b x h
            labels = tf.expand_dims(caption[:,i], 1) # b x 1
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
            concated = tf.concat(1, [indices, labels]) # b x 2
            onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0) # b x w
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i])

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b) # b x w
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels) # b x 1
            cross_entropy = cross_entropy * caption_mask[:,i] # b x 1
            loss_caption += tf.reduce_sum(cross_entropy) # 1

        loss_caption = loss_caption / tf.reduce_sum(caption_mask)
        loss = loss_caption
        return loss, video, video_mask, caption, caption_mask


    def build_generator(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])
        image_emb = tf.transpose(image_emb, [1,0,2])

        state1 = tf.zeros([self.batch_size, self.lstm3.state_size])
        h_prev = tf.zeros([self.batch_size, self.dim_hidden])

        generated_words = []

        current_embed = tf.zeros([self.batch_size, self.dim_hidden])
        brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_lstm_steps,1,1])   # n x h x 1
        image_part = tf.batch_matmul(image_emb, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.n_lstm_steps,1,1])) +  self.embed_att_ba # n x b x h
        for i in range(n_caption_step):
            e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part) # n x b x h
            e = tf.batch_matmul(e, brcst_w)
            e = tf.reduce_sum(e,2) # n x b
            e_hat_exp = tf.mul(tf.transpose(video_mask), tf.exp(e)) # n x b
            denomin = tf.reduce_sum(e_hat_exp,0) # b
            denomin = denomin + tf.to_float(tf.equal(denomin, 0))
            alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h
            attention_list = tf.mul(alphas, image_emb) # n x b x h                
            atten = tf.reduce_sum(attention_list,0) # b x h

            if i > 0: tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM3") as vs:
                output1, state1 = self.lstm3( tf.concat(1,[atten, current_embed]), state1 ) # b x h
                lstm3_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

            output2 = tf.tanh(tf.nn.xw_plus_b(tf.concat(1,[output1,atten,current_embed]), self.embed_nn_Wp, self.embed_nn_bp)) # b x h
            h_prev = output1
            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b) # b x w
            max_prob_index = tf.argmax(logit_words, 1) # b
            generated_words.append(max_prob_index) # b
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)

        generated_words = tf.transpose(tf.pack(generated_words))
        return video, video_mask, generated_words, lstm3_variables


############### Global Parameters ###############
video_data_path_train = '/home/PaulChen/h5py_data/cont_augment/train_vn.txt'
video_data_path_val = '/home/PaulChen/h5py_data/cont_augment/val.txt'
video_data_path_test = '/home/PaulChen/h5py_data/cont_augment/test.txt'
video_feat_path = '/home/PaulChen/h5py_data/cont_augment/'

model_path = '/home/PaulChen/evalmodel/Att_baseline/models'
############## Train Parameters #################
dim_image = 4096*2
dim_hidden= 512*2
n_frame_step = 45
n_caption_step = 35
n_epochs = 200
batch_size = 100
learning_rate = 0.0001 
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
    train_title = []
    title = []
    fname = []
    for ele in video_list_train:
        batch_data = h5py.File(ele)
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in xrange(len(batch_fname)):
                fname.append(batch_fname[i])
                title.append(batch_title[i])
                train_title.append(batch_title[i])

    video_list_val = get_video_data_HL(video_data_path_val, video_feat_path)
    for ele in video_list_val:
        batch_data = h5py.File(ele)
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in xrange(len(batch_fname)):
                fname.append(batch_fname[i])
                title.append(batch_title[i])

    video_list_test = get_video_data_HL(video_data_path_test, video_feat_path)
    for ele in video_list_test:
        batch_data = h5py.File(ele)
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in xrange(len(batch_fname)):
                fname.append(batch_fname[i])
                title.append(batch_title[i])

    fname = np.array(fname)
    title = np.array(title)
    train_title = np.array(train_title)
    video_data = pd.DataFrame({'Description':train_title})

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

def testing_one(sess, video_feat_path, ixtoword, video_tf, video_mask_tf, caption_tf, counter):
    pred_sent = []
    gt_sent = []
    IDs = []
    namelist = []
    #print video_feat_path
    test_data_batch = h5py.File(video_feat_path)
    gt_captions = json.load(open('msvd2sent.json'))

    video_feat = np.zeros((batch_size, n_frame_step, dim_image))
    video_mask = np.zeros((batch_size, n_frame_step))
#    video_feat = np.transpose(test_data_batch['data'],[1,0,2])
    
    for ind in xrange(batch_size):
        video_feat[ind,:,:] = test_data_batch['data'][:n_frame_step,ind,:]
        idx = np.where(test_data_batch['label'][:,ind] != -1)[0]
        if(len(idx) == 0):
            continue
        video_mask[ind,:idx[-1]+1] = 1.
    
    generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
    #ipdb.set_trace()

    for ind in xrange(batch_size):
        cap_key = test_data_batch['fname'][ind]
        if cap_key == '':
            break
        else:
            generated_words = ixtoword[generated_word_index[ind]]
            punctuation = np.argmax(np.array(generated_words) == '.')+1
            generated_words = generated_words[:punctuation]
            #ipdb.set_trace()
            generated_sentence = ' '.join(generated_words)
            pred_sent.append([{'image_id':str(counter),'caption':generated_sentence}])
            namelist.append(cap_key)
            for i,s in enumerate(gt_captions[cap_key]):
                s = unicodedata.normalize('NFKD', s).encode('ascii','ignore')
                gt_sent.append([{'image_id':str(counter),'cap_id':i,'caption':s}])
                IDs.append(str(counter))
            counter += 1

    return pred_sent, gt_sent, IDs, counter, namelist

def testing_all(sess, test_data, ixtoword, video_tf, video_mask_tf, caption_tf):
    pred_sent = []
    gt_sent = []
    IDs_list = []
    flist = []
    counter = 0
    gt_dict = defaultdict(list)
    pred_dict = {}
    for _, video_feat_path in enumerate(test_data):
        [b,c,d, counter, fns] = testing_one(sess, video_feat_path, ixtoword, video_tf, video_mask_tf, caption_tf, counter)
        pred_sent += b
        gt_sent += c
        IDs_list += d
        flist += fns

    for k,v in zip(IDs_list,gt_sent):
        gt_dict[k].append(v[0])

    new_flist = []
    new_IDs_list = []
    for k,v in zip(range(len(pred_sent)),pred_sent):
        if flist[k] not in new_flist:
            new_flist.append(flist[k])
            new_IDs_list.append(str(k))
            pred_dict[str(k)] = v

#pdb.set_trace()
    return pred_sent, gt_sent, new_IDs_list, gt_dict, pred_dict

def train():
    meta_data, train_data, val_data, test_data = get_video_data_jukin(video_data_path_train, video_data_path_val, video_data_path_test)
    captions = meta_data['Description'].values
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=1)

    np.save('./data0/ixtoword', ixtoword)

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate = 0.5,
            bias_init_vector=None)

    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask= model.build_model()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    with tf.device("/cpu:0"):
        saver = tf.train.Saver(max_to_keep=100)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.initialize_all_variables().run()

    tStart_total = time.time()
    for epoch in range(n_epochs):
        index = np.arange(len(train_data))
        np.random.shuffle(index)
        train_data = train_data[index]

        tStart_epoch = time.time()
        loss_epoch = np.zeros(len(train_data))
        for current_batch_file_idx in xrange(len(train_data)):

            tStart = time.time()
            current_batch = h5py.File(train_data[current_batch_file_idx])
            current_feats = np.zeros((batch_size, n_frame_step, dim_image))
            current_video_masks = np.zeros((batch_size, n_frame_step))
            current_video_len = np.zeros(batch_size)
            for ind in xrange(batch_size):
                current_feats[ind,:,:] = current_batch['data'][:n_frame_step,ind,:]
                idx = np.where(current_batch['label'][:,ind] != -1)[0]
                if len(idx) == 0:
                    continue
                current_video_masks[ind,:idx[-1]+1] = 1

            current_captions = current_batch['title']
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_step-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })
            loss_epoch[current_batch_file_idx] = loss_val
            tStop = time.time()
            #print "Epoch:", epoch, " Batch:", current_batch_file_idx, " Loss:", loss_val
        #print "Time Cost:", round(tStop - tStart,2), "s"

        print "Epoch:", epoch, " done. Loss:", np.mean(loss_epoch)
        tStop_epoch = time.time()
        print "Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s"

        if np.mod(epoch, 10) == 0 or epoch == n_epochs - 1:
            print "Epoch ", epoch, " is done. Saving the model ..."
            with tf.device("/cpu:0"):
                saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

            current_batch = h5py.File(val_data[np.random.randint(0,len(val_data))])
            video_tf, video_mask_tf, caption_tf, lstm3_variables_tf = model.build_generator()
            ixtoword = pd.Series(np.load('./data0/ixtoword.npy').tolist())
            [pred_sent, gt_sent, id_list, gt_dict, pred_dict] = testing_all(sess, train_data[-2:], ixtoword, video_tf, video_mask_tf, caption_tf)
            for key in pred_dict.keys():
                for ele in gt_dict[key]:
                    print "GT:  " + ele['caption']
                print "PD:  " + pred_dict[key][0]['caption']
                print '-------'
            [pred_sent, gt_sent, id_list, gt_dict, pred_dict] = testing_all(sess, val_data, ixtoword,video_tf, video_mask_tf, caption_tf)
            scorer = COCOScorer()
            total_score = scorer.score(gt_dict, pred_dict, id_list)
        sys.stdout.flush()

    print "Finally, saving the model ..."
    with tf.device("/cpu:0"):
        saver.save(sess, os.path.join(model_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print "Total Time Cost:", round(tStop_total - tStart_total,2), "s"

def test(model_path='models/model-900', video_feat_path=video_feat_path):
    meta_data, train_data, val_data, test_data = get_video_data_jukin(video_data_path_train, video_data_path_val, video_data_path_test)
#    test_data = val_data   # to evaluate on testing data or validation data
    ixtoword = pd.Series(np.load('./data0/ixtoword.npy').tolist())

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate = 0,
            bias_init_vector=None)

    video_tf, video_mask_tf, caption_tf, lstm3_variables_tf = model.build_generator()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    with tf.device("/cpu:0"):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
    
    for ind, row in enumerate(lstm3_variables_tf):
        if ind % 4 == 0:
                assign_op = row.assign(tf.mul(row,1-0.5))
                sess.run(assign_op)
    
    [pred_sent, gt_sent, id_list, gt_dict, pred_dict] = testing_all(sess, test_data, ixtoword,video_tf, video_mask_tf, caption_tf)
    #np.savez('Att_result/'+model_path.split('/')[1],gt = gt_sent,pred=pred_sent)
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
