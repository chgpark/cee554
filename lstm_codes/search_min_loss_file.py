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

    return min_meta
if __name__ == '__main__':
    search_min_loss_meta_file("/home/shapelim/RONet/test5/")