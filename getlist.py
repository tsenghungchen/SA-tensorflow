import glob
path = '/media/addhd4/paul/h5py_copy/msrvtt/aug-all/'
#### get the HDF5 data list 
def getlist(split):
    List = glob.glob(path+split+'*.h5')
    f = open(path+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')

    f.close()
if __name__ == '__main__':
    getlist('train')
    getlist('val')
    getlist('test')
