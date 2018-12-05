import configparser
import pathlib


class DatasetParser(object):
    """
    Handle parsing of .data file

    The .data files have the following structure:

    classes=80
    train=../coco/trainvalno5k.txt
    valid=../coco/5k.txt
    names=data/coco.names
    backup=backup/
    eval=coco
    """

    def __init__(self, data_file):
        self._data = self.parse_data_config(data_file)

        self.classes = int(self._data['classes'])
        self.train_file = self._data['train']
        self.validation_file = self._data['valid']
        self.names_file = self._data['names']
        self.backup = self._data.get('backup', 'backup')
        self.eval = self._data.get('eval')

        self._train_images = None
        self._validation_images = None
        self._train_labels = None
        self._validation_labels = None
        self._labels = None

    @property
    def train_images(self):
        """
        Read paths for train images from the train file.

        Parses train_file only once and only when requested.

        :return: paths to train image files
        """
        if self._train_images is None:
            # Extracts class labels from file
            self._train_images = self._read_file_lines(self.train_file)
        return self._train_images

    @property
    def validation_images(self):
        """
        Read paths for validation images from the validation file.

        Parses validation_file only once and only when requested.

        :return: paths to validation image files
        """
        if self._validation_images is None:
            # Extracts class labels from file
            self._validation_images = self._read_file_lines(self.validation_file)
        return self._validation_images

    @property
    def train_annotations(self):
        """
        Return paths for train images annotation files.

        Annotation file paths should be the same as the images path, replacing
        /images/ part of the path for /labels/.

        Filenames are the same as the image file name, but using txt extension.

        Returns ALL the possible annotations (some of them may not exist, p.e.
        when there are no objects in the image --negative samples--)

        :return: paths to train image annotations
        """
        if self._train_labels is None:
            self._train_labels = [
                str(
                    pathlib.Path(
                        i.replace('images/', 'labels/')
                    ).with_suffix('.txt')
                )
                for i in self.train_images
            ]
        return self._train_labels

    @property
    def validation_annotations(self):
        """
        Return paths for validation images annotation files.

        Annotation file paths should be the same as the images path, replacing
        /images/ part of the path for /labels/.

        Filenames are the same as the image file name, but using txt extension.

        Returns ALL the possible annotations (some of them may not exist, p.e.
        when there are no objects in the image --negative samples--)

        :return: paths to validation image annotations
        """
        if self._validation_labels is None:
            self._validation_labels = [
                str(
                    pathlib.Path(
                        i.replace('images/', 'labels/')
                    ).with_suffix('.txt')
                )
                for i in self.validation_images
            ]
        return self._validation_labels

    @property
    def labels(self):
        """
        Parse names file.

        :return: class name list
        """
        if self._labels is None:
            # Extracts class labels from file
            self._labels = self._read_file_lines(self.names_file)
        return self._labels

    @staticmethod
    def parse_data_config(path):
        """Parses the data configuration file"""
        with open(path, 'r') as fp:
            raw_data = fp.read()

        cp = configparser.ConfigParser()

        # ConfigParser requires at least one section. We add 'foo' manually
        cp.read_string('[foo]\n' + raw_data)

        return dict(cp['foo'].items())

    def _read_file_lines(self, file_path):
        with open(file_path, mode='r') as fp:
            return [x.strip() for x in fp.readlines()]


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs
