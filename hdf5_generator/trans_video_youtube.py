import h5py
import numpy as np
import json
import pdb
import unicodedata
import glob
import os

path = '/home/PaulChen/hdf5_generator/msvd_data/data_ch10/h5py/'
feature_folder = 'cont_captions'

def trans_video_youtube(datasplit):

    #pdb.set_trace()
    re = json.load(open('msvd2sent.json'))
    List = open(path+'cont/'+datasplit+'.txt').read().split('\n')[:-1]
    batch_size = 100
    n_length = 45

    initial = 0
    cnt = 0
    fname = []
    title = []
    data = []
    label = []
    for ele in List:
        print ele
        train_batch = h5py.File(ele)
        for idx, yyy in enumerate(train_batch['title']):
            #pdb.set_trace()
            if yyy in re.keys():
                for xxx in re[yyy]:
                    #pdb.set_trace()
                    if len(xxx.split(' ')) < 35:
                        fname.append(yyy) 
                        title.append(unicodedata.normalize('NFKD', xxx).encode('ascii','ignore'))
                        data.append(train_batch['data'][:,idx,:])
                        label.append(train_batch['label'][:,idx])
                        cnt += 1
                        if cnt == batch_size:
                            batch = h5py.File(path+feature_folder+'/'+datasplit+str(initial)+'.h5','w')
                            data = np.transpose(data,(1,0,2))
                            batch['data'] = np.array(data)#np.zeros((n_length,batch_size,4096*2))
                            fname = np.array(fname)
                            title = np.array(title)
                            batch['fname'] = fname
                            batch['title'] = title
#                           batch['pos'] = np.zeros(batch_size)
                            batch['label'] = np.transpose(np.array(label))#np.zeros((n_length,batch_size))
                            fname = []
                            title = []
                            label = []
                            data = []
                            cnt = 0
                            initial += 1
        if ele == List[-1] and len(fname) > 0:
            while len(fname) < batch_size:
                fname.append('')
                title.append('')
            batch = h5py.File(path+feature_folder+'/'+datasplit+str(initial)+'.h5','w')
            batch['data'] = np.zeros((n_length,batch_size,4096*2))
            batch['data'][:,:len(data),:] = np.transpose(np.array(data),(1,0,2))#np.zeros((n_length,batch_size,4096*2))
            fname = np.array(fname)
            title = np.array(title)
            batch['fname'] = fname
            batch['title'] = title
            batch['label'] = np.ones((n_length,batch_size))*(-1)
            batch['label'][:,:len(data)] = np.array(label).T
#            batch['pos'] = np.zeros(batch_size)
#            batch['label'] = np.zeros((n_length,batch_size))



def getlist(feature_folder_name, split):
    list_path = os.path.join(path, feature_folder_name+'/')
    List = glob.glob(list_path+split+'*.h5')
    f = open(list_path+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')


if __name__ == '__main__':
    trans_video_youtube('train')
    trans_video_youtube('val')
    trans_video_youtube('test')
    getlist(feature_folder,'train')
    getlist(feature_folder,'val')
    getlist(feature_folder,'test')

