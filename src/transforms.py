

from torchvision import transforms


class Transforms_Train:
    """Class that contains the transformations to be applied to the images."""
    
    def __init__(self):
        """Initializes the transformations."""
        self.transformer = transforms.Compose([
            transforms.CenterCrop(size=(46, 46)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __call__(self, img):
        return self.transformer(img)



class Transforms_Test:
    """Class that contains the transformations to be applied to the images."""
    
    def __init__(self):
        """Initializes the transformations."""
        self.transformer = transforms.Compose([
            transforms.CenterCrop(size=(46, 46)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __call__(self, img):
        return self.transformer(img)