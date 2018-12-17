import argparse
import os
import cv2

import matplotlib.pyplot as plt
from matplotlib import patches
from torchvision.transforms import transforms

from utils import parsers
from utils import datasets
from utils import transforms as transf
from utils.utils import *


def get_annotation_rectangle(image, annotation):
    h, w = image.shape[:2]  # shape = [height, width]
    width = int(annotation[3]*w)
    height = int(annotation[4]*h)
    x, y = (
        int(annotation[1]*w - width/2),
        int(annotation[2]*h - height/2),
    )
    return x, y, width, height


def draw_annotation_in_image(image, annotation, **kwargs):
    kwargs.setdefault('color', (0, 255, 0))
    kwargs.setdefault('thickness', 2)
    x, y, w, h = get_annotation_rectangle(image, annotation)
    top_left_corner = (x, y)
    bottom_right_corner = (x + w, y + h)
    return cv2.rectangle(
        image,
        top_left_corner,
        bottom_right_corner,
        **kwargs)


def show_objects(image, objects):
    """Show image with landmarks"""
    plt.imshow(image)

    h, w = image.shape[:2]  # shape = [height, width]

    # Get the current reference
    ax = plt.gca()

    for ann in objects:
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (
                (ann[1]*w - ann[3]*w/2),
                (ann[2]*h - ann[4]*h/2),
            ),
            ann[3]*w,
            ann[4]*h,
            linewidth=1,
            edgecolor='r',
            facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--data-config', type=str, default='cfg/coco.data', help='path to data config file')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')

    parser.add_argument('--plot', action='store_true', help='plot images')
    parser.add_argument('--save', action='store_true', help='save images')

    parser.add_argument('--output', type=str, default='output', help='path to store output images')

    opt = parser.parse_args()

    if not any([opt.plot, opt.save]):
        parser.error("At least one of --plot or --save must be given")

    print(opt, end='\n\n')

    init_seeds()

    data = parsers.DatasetParser(opt.data_config)

    transform_data = transforms.Compose([
        transf.ToPILImage(),
        transf.ToRGB(),
        transf.PadToSquare(),
        transf.Rescale(opt.img_size),
        transf.ToNumpy(),
    ])

    face_dataset = datasets.AnnotatedDataset(
        data.train_images,
        data.train_annotations,
        transform=transform_data,
        root_dir=None)

    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample = face_dataset[i]

        print(i, face_dataset.data[i], sample.sample.shape, len(sample.annotations))

        image = sample.sample
        for ann in sample.annotations:
            image = draw_annotation_in_image(image, ann)

        if opt.save:
            output_image_filename = os.path.join(
                opt.output,
                sample.sample_filename,
            )
            cv2.imwrite(output_image_filename, image)

        if opt.plot:
            fig, ax = plt.subplots()
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            plt.imshow(image)
            plt.pause(0.001)

            key = input('Press enter to continue: ')

            if key == 'q':
                break
            else:
                pass

        # save image
