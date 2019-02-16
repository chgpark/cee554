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

def search_meta_file_list(dirname):
    meta_file_list = []
    filenames = os.listdir(dirname)
    for i,filename in enumerate(filenames):
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.meta':
            meta_name = os.path.splitext(full_filename)[0]
            meta_file_list.append(meta_name)

    return meta_file_list
if __name__ == '__main__':
    print(search_min_loss_meta_file("/home/shapelim/RONet/0209_3/"))
    print(search_min_loss_meta_file("/home/shapelim/RONet/0209_3/")[-7:])
    print(search_meta_file_list("/home/shapelim/RONet/0209_3/"))
