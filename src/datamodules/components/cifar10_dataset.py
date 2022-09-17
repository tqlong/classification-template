from torchvision.datasets import CIFAR10


class CIFAR10Dataset(CIFAR10):
    def __init__(self, root, train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index: int):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label

