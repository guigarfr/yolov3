import argparse

import matplotlib.pyplot as plt
from matplotlib import patches
from torchvision.transforms import transforms

from utils import parsers
from utils import datasets
from utils import transforms as transf
from utils.utils import *


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
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    data = parsers.DatasetParser(opt.data_config)

    transform_data = transforms.Compose([
        transf.ToNumpy(),
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

        print(i, face_dataset.data[i], sample['image'].shape, len(sample['objects']))

        fig, ax = plt.subplots()
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_objects(**sample)

        key = input('Press enter to continue: ')

        if key == 'q':
            break
        else:
            pass
