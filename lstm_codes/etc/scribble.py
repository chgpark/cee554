import numpy as np
import os.path

def search_min_loss_meta_file(dirname):
    min_loss = 1000000000
    filenames = os.listdir(dirname)
    for i,filename in enumerate(filenames):
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.meta':
            meta_name = os.path.splitext(full_filename)[0]
            loss = int(meta_name.split('-')[0].split('_')[-1])
            if  loss < min_loss:
                min_loss = loss
                min_meta = meta_name

    print (min_meta)
if __name__ == '__main__':
    # search_min_loss_meta_file("/home/shapelim/RONet/test5")
    txt = np.loadtxt("/home/shapelim/RONet/val_Karpe_181102/1103_Karpe_test2.csv", delimiter=',')
    print (txt)
    position = txt[:,-3:]
    print (position)
    rounded_position = np.round(position/0.1)*0.1
    print(rounded_position)
    all_txt = np.concatenate((txt[:,:-3], rounded_position), axis = 1)
    print (all_txt)
    print ("hi!")