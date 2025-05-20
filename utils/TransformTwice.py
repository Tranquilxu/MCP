from utils.randaugment import RandAugmentMC
from torchvision import transforms

def convert_to_rgb(image):
    return image.convert("RGB")


class TransformTwice:
    #  cifar100
    def __init__(self, transform, test=0, image_size=224):
        self.owssl = transform
        self.test = test

        self.simclr = transforms.Compose([
            convert_to_rgb,
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.fixmatch_weak = transforms.Compose([
            convert_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.fixmatch_strong = transforms.Compose([
            convert_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, inp):
        out1 = self.owssl(inp)
        out4 = self.owssl(inp)
        # out3 = self.simclr(inp)
        # out4 = self.simclr(inp)
        out2 = self.fixmatch_strong(inp)
        out3 = self.fixmatch_weak(inp)
        if self.test == 0:
            return out1, out2, out3
        if self.test == 1:
            return out1, out4