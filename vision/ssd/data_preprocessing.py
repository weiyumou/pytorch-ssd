from ..transforms.transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """

        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            # lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            self._get_img_box_label,
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)

    def _get_img_box_label(self, img, boxes=None, labels=None):
        return img / self.std, boxes, labels


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.std = std
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            # lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            self._get_img_box_label,
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)

    def _get_img_box_label(self, img, boxes=None, labels=None):
        return img / self.std, boxes, labels


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image
