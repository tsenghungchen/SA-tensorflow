import os
import cv2
import json
import pdb
import numpy as np

video_path = '/media/data/MSVD/YouTubeClips'
output_path = 'msvd_data/data_ch10/labels_complete/'

# Split data to train data, valid data and test data
def splitdata(path, train_num, val_num):
    lst = os.listdir(path)
    name = []
    for ele in lst:
        name.append(os.path.splitext(ele)[0])

    print len(name)
    print name[0:100]
    name = np.random.permutation(name)
    print name[0:100]

    train = name[0:train_num]
    val = name[train_num:train_num+val_num]
    test = name[train_num+val_num:]
    np.savez('msvd_dataset',train=train, val=val, test=test)


def get_total_frame_number(fn):
    cap = cv2.VideoCapture(fn)
    if not cap.isOpened():
        print "could not open :",fn
        sys.exit() 
    length = float(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    return length

def getlist(Dir):
    pylist = []
    for root, dirs, files in os.walk(Dir):
        for ele in files:
            if ele.endswith('avi'):
                pylist.append(root+'/'+ele)

    return pylist

def get_frame_list(frame_num):
    start = 0.0
    i = 0
    end = 0.0
    frame_list = []
    if frame_num >450:
        frame_num = 450.0
    while end < frame_num:
        start = 10.0*i
        end = start + 10.0
        i += 1
        if end > frame_num:
            end = frame_num
#	if end - start >= 16:
        frame_list.append([start,end])
    return frame_list

def get_label_list(fname):
#    ext = ['.3gp', '.avi', '.flv', '.m4v', '.mov', '.mp4', '.mpg', '.mts', '.wmv','.mpeg','.mkv']
#    for e in ext:
#        file = video_path+fname+e
#    for file in filelist:
#        if os.path.isfile(file):
#            print file
#            break
    
    frame_len = get_total_frame_number(fname)
    frame_list = get_frame_list(frame_len)
    label_list = [-1]*len(frame_list)
    label_list[-1] = 0
    fname = fname.split('/')[-1].split('.')[0]
    outfile = output_path+str(fname)+'.json'
#    print outfile
    if not os.path.isfile(outfile):
    	json.dump([frame_list, label_list], open(outfile,"w"))

if __name__=='__main__':
    b = getlist(video_path)
    print b
    count = 0
    for ele in b:
        fname = ele
        if not os.path.isfile(output_path+str(fname)+'.json'):
            get_label_list(fname)
        count += 1
    print count

