from scipy.misc import imread, imresize
# import cv2
import os
import numpy as np
import cPickle
import time
from progressbar import ProgressBar

class imdb(object):
    def __init__(self, name, size_image=64, image_value_range=(-1,1), num_categories=10):
        self._name = name
        self._root_dir = None
        self._train_ind = None
        self._test_ind = None
        self._size_image = size_image
        self._image_value_range = image_value_range
        self._num_categories = num_categories

    @property
    def name(self):
        return self._name

    @property
    def train_ind(self):
        return self._train_ind

    @property
    def test_ind(self):
        return self._test_ind

    @property
    def size_image(self):
        return self._size_image

    @property
    def image_value_range(self):
        return self._image_value_range

    @property
    def num_categories(self):
        return self._num_categories

    def load_image(
        self,
        image_path,
        image_size=64,
        image_value_range=(-1, 1),
        is_gray=False,
    ):
        if is_gray:
            # image = cv2.imread(os.path.join('./data', image_path), 0).astype(np.float32)
            image = imread(os.path.join('./data', image_path), mode='L').astype(np.float32)
        else:
            # image = cv2.imread(os.path.join('./data', image_path), 1).astype(np.float32)
            image = imread(os.path.join('./data', image_path), mode='RGB').astype(np.float32)
        # im_scale_x = float(image_size) / float(image.shape[1])
        # im_scale_y = float(image_size) / float(image.shape[0])
        # image = cv2.resize(image, None, None, fx=im_scale_x, fy=im_scale_y,
        #                     interpolation=cv2.INTER_LINEAR)
        image = imresize(image, [image_size, image_size])
        image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
        return image


class UTKFace(imdb):
    def __init__(self, size_image=64, image_value_range=(-1,1), num_categories=10, in_memory=False):
        super(UTKFace, self).__init__('UTKFace', size_image, image_value_range, num_categories)
        self._root_dir = os.path.join('./data', self.name)
        self._info = self._load_info()                  ## file_name in order
        self._namelist_ind = self._load_ind()           ## ind -> filename
        self._train_ind = self._namelist_ind[:-100]
        self._test_ind = self._namelist_ind[-100:]
        self._save_test_pkl()
        self._in_memory = in_memory
        if self._in_memory:
            print('\tLoading into memory ...')
            self._imgs = self._load_imgs()              ## uint8
            self._name2ind = self._get_name2ind()

    def _load_info(self):
        info_path = sorted(os.listdir(self._root_dir))
        return [os.path.join(self.name, item) for item in info_path]

    def _load_ind(self):
        ind = range(len(self._info))
        np.random.shuffle(ind)
        return ind

    def _load_imgs(self):
        ret = []
        pbar = ProgressBar(maxval=len(self._info))
        pbar.start()
        i = 0
        for i, item in enumerate(self._info):
            ret.append(imread(os.path.join('./data', item), mode='RGB'))
            pbar.update(i)
        pbar.finish()
        return ret

    def _get_name2ind(self):
        ret = {}
        for i in range(len(self._info)):
            ret[self._info[i]] = i
        return ret

    def _save_test_pkl(self):
        cache_file = os.path.join('./data', 'UTK_test_files.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(self._test_ind, fid, cPickle.HIGHEST_PROTOCOL)

    def get_dataset(self, sample_inds):
        if self._in_memory:
            sample = []
            for sample_ind in sample_inds:
                image = self._imgs[self._name2ind[self._info[sample_ind]]].astype(np.float32)
                image = imresize(image, [self.size_image, self.size_image])
                image = image.astype(np.float32) * (self.image_value_range[-1] - self.image_value_range[0]) / 255.0 + self.image_value_range[0]
                sample.append(image)
        else:
            sample = [self.load_image(
                image_path=self._info[sample_ind],
                image_size=self.size_image,
                image_value_range=self.image_value_range,
                is_gray=False,
            ) for sample_ind in sample_inds]
        sample_images = np.array(sample).astype(np.float32)

        sample_label_age = np.ones(
            shape=(len(sample_inds), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        sample_label_gender = np.ones(
            shape=(len(sample_inds), 2),
            dtype=np.float32
        ) * self.image_value_range[0]

        for i, ind in enumerate(sample_inds):
            sample_file = self._info[ind]
            label = int(str(sample_file).split('/')[-1].split('_')[0])
            if 0 <= label <= 5:
                label = 0
            elif 6 <= label <= 10:
                label = 1
            elif 11 <= label <= 15:
                label = 2
            elif 16 <= label <= 20:
                label = 3
            elif 21 <= label <= 30:
                label = 4
            elif 31 <= label <= 40:
                label = 5
            elif 41 <= label <= 50:
                label = 6
            elif 51 <= label <= 60:
                label = 7
            elif 61 <= label <= 70:
                label = 8
            else:
                label = 9
            sample_label_age[i, label] = self.image_value_range[-1]
            gender = int(str(sample_file).split('/')[-1].split('_')[1])
            sample_label_gender[i, gender] = self.image_value_range[-1]
        return sample_images, sample_label_age, sample_label_gender


class IMDBWIKI(imdb):
    def __init__(self, size_image=64, image_value_range=(-1,1), num_categories=6):
        super(IMDBWIKI, self).__init__('IMDBWIKI', size_image, image_value_range, num_categories)
        self._info = self._load_info()              ## age, path, gender
        self._namelist_ind = self._load_namelist_ind()      ## ind_list instead of filename_list
        self._train_ind = self._namelist_ind[:100000]
        self._test_ind = self._namelist_ind[-100:]
        self._save_test_pkl()

    def _load_info(self):
        info_path = os.path.join('./data', 'imdb_wiki.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as fid:
                return cPickle.load(fid)
        else:
            raise Exception("{} not find".format(info_path))

    def _load_namelist_ind(self):
        ind = range(len(self._info))
        np.random.shuffle(ind)
        return ind

    def _save_test_pkl(self):
        cache_file = os.path.join('./data', 'IW_test_files.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(self._test_ind, fid, cPickle.HIGHEST_PROTOCOL)

    def get_dataset(self, sample_inds):
        # st = time.time()
        sample = [self.load_image(
            image_path=self._info[sample_ind][1],
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=False,
        ) for sample_ind in sample_inds]
        # print(time.time()-st)
        sample_images = np.array(sample).astype(np.float32)

        sample_label_age = np.ones(
            shape=(len(sample_inds), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        sample_label_gender = np.ones(
            shape=(len(sample_inds), 2),
            dtype=np.float32
        ) * self.image_value_range[0]

        for i, ind in enumerate(sample_inds):
            label, _, gender = self._info[ind]
            label = int(label)
            gender = int(gender)
            if gender == -1:
                gender = np.random.randint(2)
            if 0 <= label <= 18:
                label = 0
            elif 19 <= label <= 29:
                label = 1
            elif 30 <= label <= 39:
                label = 2
            elif 40 <= label <= 49:
                label = 3
            elif 50 <= label <= 59:
                label = 4
            else:
                label = 5
            sample_label_age[i, label] = self.image_value_range[-1]
            sample_label_gender[i, gender] = self.image_value_range[-1]
        return sample_images, sample_label_age, sample_label_gender

                       