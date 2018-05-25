from scipy.misc import imread, imresize, imsave
import os
import numpy as np
import cPickle

class imdb(object):
    def __init__(self, name, size_image=64, image_value_range=(-1,1), num_categories=10):
        self._name = name
        self._root_dir = None
        self._train_list = None
        self._test_list = None
        self._size_image = size_image
        self._image_value_range = image_value_range
        self._num_categories = num_categories

    @property
    def name(self):
        return self._name

    @property
    def train_list(self):
        return self._train_list

    @property
    def test_list(self):
        return self._test_list

    @property
    def size_image(self):
        return self._size_image

    @property
    def image_value_range(self):
        return self._image_value_range

    @property
    def num_categories(self):
        return self._num_categories

    def get_dataset(self, sample_files):
        raise NotImplementedError

    def load_image(
        self,
        image_path,
        image_size=64,
        image_value_range=(-1, 1),
        is_gray=False,
    ):
        if is_gray:
            image = imread(os.path.join('./data', image_path), mode='L').astype(np.float32)
        else:
            image = imread(os.path.join('./data', image_path), mode='RGB').astype(np.float32)
        image = imresize(image, [image_size, image_size])
        image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
        return image


class UTKFace(imdb):
    def __init__(self, size_image=64, image_value_range=(-1,1), num_categories=10):
        super(UTKFace, self).__init__('UTKFace', size_image, image_value_range, num_categories)
        self._root_dir = os.path.join('./data', self.name)
        self._namelist = self._load_namelist()
        self._train_list = self._namelist[:-100]
        self._test_list = self._namelist[-100:]
        self._save_test_pkl()

    def _load_namelist(self):
        file_names = sorted(os.listdir(self._root_dir))
        np.random.shuffle(file_names)
        return [os.path.join(self.name, item) for item in file_names]

    def _save_test_pkl(self):
        cache_file = os.path.join('./data', 'UTK_test_files.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(self._test_list, fid, cPickle.HIGHEST_PROTOCOL)

    def get_dataset(self, sample_files):
        sample = [self.load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=False,
        ) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        sample_label_age = np.ones(
            shape=(len(sample_files), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        sample_label_gender = np.ones(
            shape=(len(sample_files), 2),
            dtype=np.float32
        ) * self.image_value_range[0]

        for i, label in enumerate(sample_files):
            label = int(str(sample_files[i]).split('/')[-1].split('_')[0])
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
            gender = int(str(sample_files[i]).split('/')[-1].split('_')[1])
            sample_label_gender[i, gender] = self.image_value_range[-1]
        return sample_images, sample_label_age, sample_label_gender


class IMDBWIKI(imdb):
    def __init__(self, size_image=64, image_value_range=(-1,1), num_categories=6):
        super(IMDBWIKI, self).__init__('IMDBWIKI', size_image, image_value_range, num_categories)
        self._info = self._load_info()
        self._namelist = self._load_namelist()
        self._train_list = self._namelist[:-100]
        self._test_list = self._namelist[-100:]
        self._save_test_pkl()

    def _load_info(self):
        info_path = os.path.join('./data', 'imdb_wiki.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as fid:
                return cPickle.load(fid)
        else:
            raise Exception("{} not find".format(info_path))

    def _load_namelist(self):
        file_names = self._info.keys()
        np.random.shuffle(file_names)
        return file_names

    def _save_test_pkl(self):
        cache_file = os.path.join('./data', 'IW_test_files.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(self._test_list, fid, cPickle.HIGHEST_PROTOCOL)

    def get_dataset(self, sample_files):
        sample = [self.load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=False,
        ) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        sample_label_age = np.ones(
            shape=(len(sample_files), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        sample_label_gender = np.ones(
            shape=(len(sample_files), 2),
            dtype=np.float32
        ) * self.image_value_range[0]

        for i, name in enumerate(sample_files):
            label, gender = self._info[name]
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

                       