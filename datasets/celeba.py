from torchvision.datasets import CelebA, ImageFolder
from torchvision import transforms

def CelebADataset(root, split, img_size, center_crop):
    if split in ['train', 'test']:
        return CelebA(
            root = root,
            split = split,
            transform = transform(img_size, center_crop),
            download = False
        )
    elif split == 'whole':
        return ImageFolder(
            root = root,
            transform = transform(img_size, center_crop)
        )


def transform(img_size, center_crop):
    return transforms.Compose([
        transforms.CenterCrop(center_crop),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda_func)
    ])

def lambda_func(x):
    return 2 * x - 1.0
