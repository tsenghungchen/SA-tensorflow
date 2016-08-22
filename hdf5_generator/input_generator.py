import numpy as np
import os, json, h5py, math, pdb, glob


MAX_LEN = 45
LSTM_DIM = 512
BATCH_SIZE = 10
inp_path = '/home/PaulChen/hdf5_generator/msvd_data'
h5py_path = '/home/PaulChen/hdf5_generator/msvd_data/data_ch10/h5py'
label_path = '/home/PaulChen/hdf5_generator/msvd_data/data_ch10/labels_complete'
feature_path = '/media/data/MSVD/'
splitdataset_path = '/media/data/MSVD/msvd_dataset_final.npz'
chunk = 'ch10'


def get_max_len(path):
    lst = []
    for root, dirs, files in os.walk(path):
        for ele in files:
            if ele.endswith('json'):
                lst.append(root+'/'+ele)
    print lst
    cnt = []
    for ele in lst:
        a = json.load(open(ele))
        cnt.append(len(a[0]))    
    return max(cnt)     


def read_c3d(fn):
    f = open(fn, "rb")
    s = np.fromfile(f, dtype=np.int32)
    f = open(fn, "rb")
    f.seek(20)
    v = np.fromfile(f, dtype=np.float32)
    return s, v


def get_c3d(f_path,ftype):
    v = []
    if not os.path.exists(f_path):
        return v
    v_f = os.listdir(f_path)
    num_v = len(v_f) / 3
    for v_ele in v_f:
        if v_ele.endswith(ftype):
            [_,v_tmp] = read_c3d(os.path.join(f_path,v_ele))
            v.append(v_tmp)
    v=np.array(v)
    return v


def get_VGG(f_path,ftype):
    if not os.path.exists(f_path + '.npz'):
        return []
    v = np.load(f_path + '.npz')[ftype]
    v=np.array(v)
    return v


def check_HL_nonHL_exist(label):
    idx = len(np.where(label == 1)[0])
    idy = len(np.where(label == 0)[0])
    return idx > 0 and idy > 0


def generate_h5py(X, y, q, fname, dataset, feature_folder_name, batch_start = 0):
    dirname = os.path.join(h5py_path, feature_folder_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    num = len(np.unique(q))
    if num % BATCH_SIZE == 0:
        batch_num = int(num / BATCH_SIZE)
    else:
        batch_num = int(num / BATCH_SIZE) + 1
    q_idx = 0
    f_txt = open(os.path.join(dirname, dataset + '.txt'), 'w')
    mapping = json.load(open('msvd2sent.json'))
    for i in xrange(batch_start, batch_start + batch_num):
        train_filename = os.path.join(dirname, dataset + str(i) + '.h5')
        if os.path.isfile(train_filename):
            q_idx += BATCH_SIZE
            continue
        with h5py.File(train_filename, 'w') as f:
            f['data'] = np.zeros([MAX_LEN,BATCH_SIZE,X.shape[1]])
            #f['label'] = np.zeros([MAX_LEN,BATCH_SIZE,2])
            f['label'] = np.zeros([MAX_LEN,BATCH_SIZE]) - 1
            f['cont'] = np.zeros([MAX_LEN,BATCH_SIZE])
            #f['reindex'] = np.zeros([MAX_LEN,BATCH_SIZE])
            f['reindex'] = np.zeros(MAX_LEN)
            fname_tmp = []
            title_tmp = []
            for j in xrange(BATCH_SIZE):
                X_id = np.where(q == q_idx)[0]
                #while(len(X_id) == 0 or not check_HL_nonHL_exist(y[X_id])):
                while(len(X_id) == 0):
                    q_idx += 1
                    X_id = np.where(q == q_idx)[0]
                    if q_idx > max(q):
                        while len(fname_tmp) < BATCH_SIZE:
                            fname_tmp.append('')
                            title_tmp.append('')
                        fname_tmp = np.array(fname_tmp)
                        title_tmp = np.array(title_tmp)
                        f['fname'] = fname_tmp
                        f['title'] = title_tmp
                        f_txt.write(train_filename + '\n')
                        return
                #pdb.set_trace()
                f['data'][:len(X_id),j,:] = X[X_id,:]
                f['label'][:len(X_id),j] = y[X_id]
                #for z in xrange(len(X_id)):
                #    if y[X_id[z]] < 2:
                #        f['label'][z,j,y[X_id[z]]] = 1
                #        f['label'][z,j,abs(y[X_id[z]]-1)] = -1
                f['cont'][1:len(X_id)+1,j] = 1
                #f['reindex'][:len(X_id),j] = np.arange(len(X_id))
                f['reindex'][:len(X_id)] = np.arange(len(X_id))
                f['reindex'][len(X_id):] = len(X_id)
                fname_tmp.append(fname[q_idx])
                #title_tmp.append(' '.join(fname[q_idx].split('-')[1:]))
                title_tmp.append(fname[q_idx])
                #if q_idx == num:
                if q_idx == q[-1]:
                    while len(fname_tmp) < BATCH_SIZE:
                        fname_tmp.append('')
                        title_tmp.append('')
                    fname_tmp = np.array(fname_tmp)
                    title_tmp = np.array(title_tmp)
                    f['fname'] = fname_tmp
                    f['title'] = title_tmp
                    f_txt.write(train_filename + '\n')
                    return
                q_idx += 1
            fname_tmp = np.array(fname_tmp)
            title_tmp = np.array(title_tmp)
            f['fname'] = fname_tmp
            f['title'] = title_tmp
            #f.create_dataset('title', data=title_tmp)
        f_txt.write(train_filename + '\n')


def generate_npz(X,y,q,fname,dataset,feature_folder_name):
    dirname = os.path.join(inp_path, 'data_' + chunk, 'npz', feature_folder_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = os.path.join(dirname, dataset)
    idx = np.where(y == -1)[0]
    X = np.delete(X,idx,axis = 0)
    y = np.delete(y,idx)
    q = np.delete(q,idx)
    np.savez(filename,fv=X,label=y,q=q,fname=fname)


def get_feats_depend_on_label(label, per_f, v, idx):
    X = []
    y = []
    q = []
    for l_index in xrange(len(label[0])):
        low = int(math.ceil(label[0][l_index][0] / per_f))
        up = min(len(v), int(math.ceil(label[0][l_index][1] / per_f)))
        up_ = up
        #pdb.set_trace()
        if  low >= len(v) or low == up:
            X.append(X[-1])
        else:
            ## take the mean feature of frames in one clip
            X.append(np.mean(v[low:up,:],axis=0))
            ## random sample feature of frames in one clip
#            X.append(v[np.random.randint(low,up),:])
    
        y.append(label[1][l_index])
        q.append(idx)
    return X, y, q


def load_feats(files, ftype, dataset, feature):
    X = []
    y = []
    q = []
    fname = []
    idx = 0
    for ele in files:
        print ele, idx
        l_path = os.path.join(label_path, ele + '.json')
        label = json.load(open(l_path))
        if len(label[0]) > MAX_LEN:
            continue

        f_path = os.path.join(feature_path,feature, ele)
        if feature == 'c3d':
            v = get_c3d(f_path,'fc7-1')
            per_f = 16
            if len(v) == 0:
                continue
            [x_tmp, y_tmp, q_tmp] = get_feats_depend_on_label(label, per_f, v, idx)
        elif feature == 'VGG':
            v = get_VGG(f_path,'fc7')
            per_f = 1
            if len(v) == 0:
                continue
            [x_tmp, y_tmp, q_tmp] = get_feats_depend_on_label(label, per_f, v, idx)
        elif feature == 'cont':
            v1 = get_c3d(os.path.join(feature_path,'c3d',ele),'fc7-1')
            per_f1 = 16
            v2 = get_VGG(os.path.join(feature_path,'VGG',ele),'fc7')
            per_f2 = 1
            if len(v1) == 0 or len(v2) == 0:
                print "fuck!!"
                continue
            [x1_tmp, y1_tmp, q1_tmp] = get_feats_depend_on_label(label, per_f1, v1, idx)
            [x2_tmp, y2_tmp, q2_tmp] = get_feats_depend_on_label(label, per_f2, v2, idx)
            x_tmp = map(list, zip(*(zip(*x1_tmp) + zip(*x2_tmp))))
            y_tmp = y1_tmp
            q_tmp = q1_tmp
        X += x_tmp
        y += y_tmp
        q += q_tmp
        #pdb.set_trace()
        fname.append(ele)
        idx += 1
    return np.array(X), np.array(y), np.array(q), np.array(fname)


def Normalize(X, normal = 0):
    if normal == 0:
        mean = np.mean(X,axis = 0)
        std = np.std(X,axis = 0)
        idx = np.where(std == 0)[0]
        std[idx] = 1
    else:
        mean = normal[0]
        std = normal[1]
    X = (X - mean) / std
    return X, mean, std


def driver(inp_type, f_type, Rep_type, outp_folder_name):
    dataset = 'train'
    #List = open(os.path.join('data_' + chunk, dataset + '_list.txt'),'r').read().split('\n')[:-1]
    List = np.load(splitdataset_path)[dataset]
    for iii in range(int(math.ceil(len(List) / 500.))):
    #for iii in range(21,29):
        [X, y, Q, fname] = load_feats(List[iii*500:min(len(List),(iii+1)*500)], f_type, dataset, Rep_type)
        #[X, mean, std] = Normalize(X)
        if inp_type == 'h5py':
            generate_h5py(X, y, Q, fname, dataset, outp_folder_name, batch_start = iii*50)
        else:
            generate_npz(X,y,Q, fname, dataset, outp_folder_name)    
    
    dataset = 'val'
    #List = open(os.path.join('data_'+ chunk, dataset + '_list.txt'),'r').read().split('\n')[:-1]
    List = np.load(splitdataset_path)[dataset]
    for iii in range(int(math.ceil(len(List) / 500.))):
        [X, y, Q, fname] = load_feats(List[iii*500:min(len(List),(iii+1)*500)], f_type, dataset, Rep_type)
        #[X, mean, std] = Normalize(X, [mean, std])
        if inp_type == 'h5py':
            generate_h5py(X, y, Q, fname, dataset, outp_folder_name, batch_start = iii*50)
        else:
            generate_npz(X,y,Q, fname, dataset,outp_folder_name)
    
    dataset = 'test'
    #List = open(os.path.join('data_'+ chunk, dataset + '_list.txt'),'r').read().split('\n')[:-1]
    List = np.load(splitdataset_path)[dataset]
    for iii in range(int(math.ceil(len(List) / 500.))):
    #for iii in range(3,4):
        [X, y, Q, fname] = load_feats(List[iii*500:min(len(List),(iii+1)*500)], f_type, dataset, Rep_type)
        #[X, mean, std] = Normalize(X, [mean, std])
        if inp_type == 'h5py':
            generate_h5py(X, y, Q, fname, dataset, outp_folder_name, batch_start = iii*50)
        else:
            generate_npz(X,y,Q, fname, dataset,outp_folder_name)
        

def getlist(path, split):
    List = glob.glob(path+split+'*.h5')
    print path+split+'.txt'
    f = open(path+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')

    
if __name__ == '__main__':
#    MAX_LEN = get_max_len(inp_path)
#    print MAX_LEN
    ## concatenate
    driver('h5py','fc7-1','cont','cont')

    path = os.path.join(h5py_path, 'cont' + '/')
    getlist(path,'train')
    getlist(path,'val')
    getlist(path,'test')

