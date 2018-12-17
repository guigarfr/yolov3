import numpy as np

import torch

from torchvision.transforms import functional as transform_f
from torchvision.transforms import transforms as transform_o

from PIL.Image import Image


class AnnotationTransform(object):

    def __call__(self, sample):
        image, annotations = sample.sample, sample.annotations
        t_image, t_annotations = self.transform(image, annotations)
        sample.sample = t_image
        sample.annotations = t_annotations
        return sample

    def transform(self, image, annotations):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class ToRGB(AnnotationTransform):
    """Convert ndarrays in sample to Tensors."""

    def transform(self, image, annotations):
        rgb_image = image.convert('RGB')
        return rgb_image, annotations


class PadToSquare(AnnotationTransform):
    """Pad the given PIL Image so the result image is squared."""

    def transform(self, image, annotations):

        original_size = image.size
        result_size = 2*[max(original_size)]
        padding_wh = np.subtract(result_size, original_size)
        padding_wh_2 = padding_wh // 2
        padding_wh_0 = padding_wh - padding_wh_2
        padding_wh_1 = padding_wh - padding_wh_0
        padding = (
            padding_wh_0[0],  # left
            padding_wh_0[1],  # top
            padding_wh_1[0],  # right
            padding_wh_1[1],  # bottom
        )
        
        padded_image = transform_f.pad(image, padding)

        w, h = padded_image.size
        assert w == h
        assert np.all(np.equal(padded_image.size, result_size))

        # Load labels
        padded_annotations = self.pad_annotations(
            annotations,
            original_size,
            result_size,
            padding)

        return padded_image, padded_annotations

    def pad_annotations(self, annotations, old_size, new_size, padding):
        left, top, right, bottom = padding

        if annotations is not None and annotations.any():
            assert annotations.shape[1] == 5

            padded_annotation = annotations.copy()
            padded_annotation[:, 1] = (annotations[:, 1] * old_size[0] + left)
            padded_annotation[:, 2] = (annotations[:, 2] * old_size[1] + top)
            padded_annotation[:, 3] = (annotations[:, 3] * old_size[0])
            padded_annotation[:, 4] = (annotations[:, 4] * old_size[1])

            padded_annotation[:, (1, 3)] /= new_size[0]
            padded_annotation[:, (2, 4)] /= new_size[1]
        else:
            padded_annotation = annotations

        return padded_annotation.tolist()


class Rescale(AnnotationTransform):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def transform(self, image, annotations):
        img = transform_f.resize(image, self.output_size)
        return img, annotations


class ToTensor(AnnotationTransform, transform_o.ToTensor):
    """Convert ndarrays in sample to Tensors."""

    def transform(self, image, annotations):
        image = transform_o.ToTensor.__call__(self, image).type(
            torch.FloatTensor)
        return image, annotations


class ToPILImage(AnnotationTransform):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, mode=None):
        self.mode = mode

    def transform(self, image, annotations):
        if isinstance(image, Image):
            return image, annotations
        image = transform_f.to_pil_image(image, self.mode)
        return image, annotations


class ToNumpy(AnnotationTransform):
    """Convert ndarrays in sample to Tensors."""

    def transform(self, image, annotations):
        if isinstance(image, np.ndarray):
            np_image = image
        else:
            np_image = np.asarray(image, dtype=np.uint8)
            # Convert RGB to BGR (open-cv format)
            np_image = np_image[:, :, ::-1].copy()
        if len(np_image.shape) < 3:
            raise Exception("Expected higher shape")
        return np_image, annotations
