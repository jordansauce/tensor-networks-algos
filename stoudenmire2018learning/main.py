import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from typing import Optional


def show_digit(image_vec: np.ndarray, size: Optional[int]=None) -> None:
    if size is None:
        # assuming the image is square
        size = np.sqrt(data['train']['images'][0].size).astype(int)
    plt.imshow(image_vec.reshape(size, size), cmap="Greys")
    plt.show()


def downscale_image(img: np.ndarray) -> np.ndarray:
    img_size = 28
    return img.reshape(img_size, img_size)[::2,::2].reshape(-1,1)


def load_mnist(data_path: str,
               downscale: Optional[bool] = True) -> np.ndarray:
    num_labels = 10
    img_size = 28

    # data files
    train_data_path = os.path.join(data_path, 'mnist_train.csv')
    test_data_path = os.path.join(data_path, 'mnist_test.csv')

    # loading data
    train_data = np.loadtxt(train_data_path, delimiter=',')
    test_data = np.loadtxt(test_data_path, delimiter=',')

    # scaling to [0.01, 0.99]; we want to avoid zeros and ones
    frac = 0.99 / 255

    # extracting data and labels
    train_images = np.asfarray(train_data[:, 1:]) * frac + 0.01
    train_labels = np.asfarray(train_data[:, :1])

    test_images = np.asfarray(test_data[:, 1:]) * frac + 0.01
    test_labels = np.asfarray(test_data[:, :1])

    # transform labels into one hot representation
    lr = np.arange(num_labels)
    train_labels_one_hot = (lr==train_labels).astype(np.float)
    test_labels_one_hot = (lr==test_labels).astype(np.float)

    # we don't want zeroes and ones in the labels either
    train_labels_one_hot[train_labels_one_hot==0] = 0.01
    train_labels_one_hot[train_labels_one_hot==1] = 0.99
    test_labels_one_hot[test_labels_one_hot==0] = 0.01
    test_labels_one_hot[test_labels_one_hot==1] = 0.99

    if downscale:
        train_images = np.apply_along_axis(downscale_image, 1, train_images)
        test_images = np.apply_along_axis(downscale_image, 1, test_images)

    return {
        'train': {
            'images': train_images,
            'labels': train_labels_one_hot
        },
        'test': {
            'images': test_images,
            'labels': test_labels_one_hot
        }
    }



def feature_map(x: np.ndarray) -> np.ndarray:
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mnist-data-dir', help='MNIST data path', required=True)
    args = parser.parse_args()

    data = load_mnist(args.mnist_data_dir)
    import pdb; pdb.set_trace()
