import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def augment_image(image_path, boxes, labels):
    image = np.array(Image.open(image_path).convert("RGB"))
    augmented = transform(image=image, bboxes=boxes, labels=labels)
    return augmented['image'], augmented['bboxes'], augmented['labels']

if __name__ == "__main__":
    img_path = 'data/train/img1.jpg'
    boxes = [[34, 50, 120, 160]]  # Example box
    labels = [1]
    img, new_boxes, new_labels = augment_image(img_path, boxes, labels)
    Image.fromarray(img).show()
    print(new_boxes, new_labels)
