from collections import Counter
import numpy as np
import datetime as dt
import scipy.io as sio
import cPickle
import os
from imageio import imread, imwrite
from scipy.misc import imresize
from progressbar import ProgressBar
import shutil

def birthday(t):
    try:
        a = (dt.datetime.fromordinal(int(t)) + dt.timedelta(days=t%1) - dt.timedelta(days = 366)).year
    except:
        print t
    return a

def mat2cpickle(file_name, data_name, ret):
    wiki = sio.loadmat(file_name)
    info = wiki[data_name][0][0]

    dob = info[0][0]
    res = []
    for i in range(dob.shape[0]):
        try:
            t = dob[i]
            res.append((dt.datetime.fromordinal(int(t)) + dt.timedelta(days=t%1) - dt.timedelta(days = 366)).year)
        except:
            res.append(9999)
    dob = np.array(res)
        
    photo_taken = info[1][0]
    age = photo_taken - dob
    del1 = set(np.where(age<0)[0])

    full_path = [data_name+'_crop/'+item[0] for item in info[2][0]] # list
    gender = info[3][0]
    del2 = set(np.where(np.isnan(gender))[0])

    face_location = info[5][0]
    face_score = info[6][0]
    del3 = set(np.where(face_score<1.0)[0])

    second_face_score = info[7][0]
    keep = set(np.where(np.isnan(second_face_score))[0])
    ind = list(keep - (del1 | del2 | del3))
    
    temp = zip(age, full_path, gender, face_location)
    ret.extend([temp[i] for i in ind])
    print "{} length: {}".format(data_name, len(ind))

def imdb_wiki_preprocess():
    ################# mat -> pkl #################
    # ret = []
    # file_name = '/home/sy/code/face/Face-Aging/data/wiki_crop/wiki.mat'
    # mat2cpickle(file_name, 'wiki', ret)
    # file_name = '/home/sy/code/face/Face-Aging/data/imdb_crop/imdb.mat'
    # mat2cpickle(file_name, 'imdb', ret)
    # print "total length: {}".format(len(ret))
    # cache_file = '/home/sy/code/face/Face-Aging/data/imdb_wiki.pkl'
    # with open(cache_file, 'wb') as fid:
    #     cPickle.dump(ret, fid, cPickle.HIGHEST_PROTOCOL)

    ################# resize image: wiki_crop -> wiki_crop_ #################
    cache_path = '/home/sy/code/face/Face-Aging/data/imdb_wiki.pkl'
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as fid:
            info = cPickle.load(fid)
    path = '/home/sy/code/face/Face-Aging/data/'
    dst_path = '/home/sy/code/face/Face-Aging/data/IW'
    pbar = ProgressBar(maxval=len(info))
    pbar.start()
    i = 0
    ret = []
    for i, item in enumerate(info):
        name = item[1]
        dir1, dir2, filename = name.split('/')
        image = imread(os.path.join(path, dir1, dir2, filename)).astype(np.float32)

        if image.shape[0]<128 or image.shape[1]<128:
            continue
        image = imresize(image, [128, 128])
        imwrite(os.path.join(dst_path, filename), image)
        ret.append((item[0], 'IW/' + filename, item[2]))
        i += 1
        pbar.update(i)
    pbar.finish()
    cache_file = '/home/sy/code/face/Face-Aging/data/imdb_wiki_.pkl'
    with open(cache_file, 'wb') as fid:
        cPickle.dump(ret, fid, cPickle.HIGHEST_PROTOCOL)

    ################# move to one category, change path info #################
    # cache_path = '/home/sy/code/face/Face-Aging/data/imdb_wiki.pkl'
    # if os.path.exists(cache_path):
    #     with open(cache_path, 'rb') as fid:
    #         info = cPickle.load(fid)
    # path = '/home/sy/code/face/Face-Aging/data/'
    # dst_path = '/home/sy/code/face/Face-Aging/data/IW'
    # pbar = ProgressBar(maxval=len(info))
    # pbar.start()
    # i = 0
    # ret = []
    # for i, item in enumerate(info):
    #     name = item[1]
    #     dir1, dir2, filename = name.split('/')
    #     ret.append((item[0], 'IW/' + filename, item[2]))
    #     shutil.copy(os.path.join(path, dir1, dir2, filename), dst_path)
    #     pbar.update(i)
    # pbar.finish()
    # cache_file = '/home/sy/code/face/Face-Aging/data/imdb_wiki_.pkl'
    # with open(cache_file, 'wb') as fid:
    #     cPickle.dump(ret, fid, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    imdb_wiki_preprocess()
