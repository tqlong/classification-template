from torchvision.datasets import VOCSegmentation
import numpy as np


class PascalVOCDataset(VOCSegmentation):
    def __init__(self, root, image_set: str, download: bool, transform=None):
        super().__init__(root=root, image_set=image_set, download=download, transform=None)
        self.image_mask_transform = transform

    def __getitem__(self, index: int):
        image, mask = super().__getitem__(index)
        image = np.array(image)
        mask = np.array(mask)

        if self.image_mask_transform is not None:
            transformed = self.image_mask_transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # print("transformed", image.shape, mask.shape, type(image), type(mask), image.dtype, mask.dtype)
        
        return image, mask

if __name__ == "__main__":
    import pyrootutils
    import numpy as np

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    ds = PascalVOCDataset(root / "data", image_set="trainval", download=False, transform=None)
    image, mask = ds[0]
    # print(len(ds))
    print(np.max(mask), np.unique(mask), mask.shape)

    # sizes = np.array([list(image.size) for image, _ in ds])

    # print(np.mean(sizes, axis=0), np.std(sizes, axis=0))