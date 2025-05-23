import glob
import scipy.io
from typing import Dict, Tuple
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import random
import pickle
import os
import torch
from clip import clip
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from utils.TransformTwice import TransformTwice

device = "cuda" if torch.cuda.is_available() else "cpu"

single_template = ["a photo of a {}."]


def convert_to_rgb(image):
    return image.convert("RGB")


def get_transform(image_size=384):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


#
# def load_clip() -> Module:
#     model, _ = clip.load("ViT-L/14")
#     return model.to(device).eval()
#
#
# def load_taglist(
#         dataset: str
# ) -> Tuple[Dict]:
#     dataset_root = "./datasets/" + dataset
#
#     tag_file = dataset_root + f"/{dataset}_taglist.txt"
#
#     with open(tag_file, "r", encoding="utf-8") as f:
#         taglist_or = [line.strip() for line in f]
#     taglist = taglist_or
#
#     info = {"taglist": taglist}
#     return info


def build_clip_label_embedding(model, categories):
    # print("Creating pretrained CLIP image model")
    templates = single_template
    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        openset_label_embedding = []
        for category in categories:
            # print("category =", category)
            texts = [
                template.format(
                    processed_name(category, rm_dot=True), article=article(category)
                )
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]  # 改造句子
            texts = clip.tokenize(texts)  # tokenize
            print("texts =", texts)
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            openset_label_embedding.append(text_embedding)
        openset_label_embedding = torch.stack(openset_label_embedding, dim=1)
        if run_on_gpu:
            openset_label_embedding = openset_label_embedding.cuda()

    openset_label_embedding = openset_label_embedding.t()
    return openset_label_embedding


def load_cifar100():
    with open("./datasets/CIFAR100/train", 'rb') as f:
        data_train = pickle.load(f, encoding='latin1')
    with open("./datasets/CIFAR100/test", 'rb') as f:
        data_test = pickle.load(f, encoding='latin1')
    with open("./datasets/CIFAR100/meta", 'rb') as f:
        data_meta = pickle.load(f, encoding='latin1')
    return data_train, data_test, data_meta


def read_data_cifar_100():
    data_train, data_test, data_meta = load_cifar100()
    train_data = data_train['data'].reshape((data_train['data'].shape[0], 3, 32, 32))  # .transpose((0,1,3,2))
    test_data = data_test['data'].reshape((data_test['data'].shape[0], 3, 32, 32))  # .transpose((0,1,3,2))
    train_label = data_train["fine_labels"]
    test_label = data_test["fine_labels"]
    return train_data, train_label, test_data, test_label


def read_data_tiny_imagenet_200():
    id_dict = {}
    for i, line in enumerate(open('./datasets/tiny-imagenet-200/wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    num_classes = len(id_dict)
    cls_dic = {}
    for i, line in enumerate(open('./datasets/tiny-imagenet-200/val/val_annotations.txt', 'r')):
        a = line.split('\t')
        img, cls_id = a[0], a[1]
        cls_dic[img] = id_dict[cls_id]

    train_imgs = glob.glob("./datasets/tiny-imagenet-200/train/*/*/*.JPEG")
    test_imgs = glob.glob("./datasets/tiny-imagenet-200/val/images/*.JPEG")
    train_imgs = [img_path.replace('\\', '/') for img_path in train_imgs]
    test_imgs = [img_path.replace('\\', '/') for img_path in test_imgs]

    train_labels = [id_dict[train_img.split('/')[4]] for train_img in train_imgs]
    test_labels = [cls_dic[os.path.basename(test_img)] for test_img in test_imgs]

    return train_imgs, train_labels, test_imgs, test_labels, num_classes


def read_data_eruosat():
    id_dict = {}
    for i, line in enumerate(open('./datasets/EuroSAT/EuroSAT_taglist.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    num_classes = len(id_dict)
    EuroSAT_imgs = glob.glob("./datasets/EuroSAT/2750/*/*.jpg")
    EuroSAT_imgs = [img_path.replace('\\', '/') for img_path in EuroSAT_imgs]
    EuroSAT_labels = [id_dict[img_path.split('/')[4]] for img_path in EuroSAT_imgs]
    EuroSAT_dataset = list(zip(EuroSAT_imgs, EuroSAT_labels))
    random.seed(0)
    random.shuffle(EuroSAT_dataset)
    train_size = int(0.7 * len(EuroSAT_dataset))
    train_set, test_set = EuroSAT_dataset[:train_size], EuroSAT_dataset[train_size:]
    train_imgs, train_labels = zip(*train_set)
    test_imgs, test_labels = zip(*test_set)

    return list(train_imgs), list(train_labels), list(test_imgs), list(test_labels), num_classes


def read_data_stanford_cars():
    id_dict = {}
    for i, line in enumerate(open('./datasets/stanford_cars/stanford_cars_taglist.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    num_classes = len(id_dict)
    data = scipy.io.loadmat('./datasets/stanford_cars/cars_annos.mat')
    annotations = data['annotations']
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    for i in range(annotations.shape[1]):
        name = str(annotations[0, i][0])[2:-2]
        img_path = os.path.join('./datasets/stanford_cars', name).replace('\\', '/')
        clas = int(annotations[0, i][5])
        test = int(annotations[0, i][6])
        if test == 0:
            train_imgs.append(img_path)
            train_labels.append(clas - 1)
        elif test == 1:
            test_imgs.append(img_path)
            test_labels.append(clas - 1)
    return train_imgs, train_labels, test_imgs, test_labels, num_classes


def read_data_caltech_101():
    id_dict = {}
    for i, line in enumerate(open('./datasets/caltech-101/caltech-101_taglist.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    num_classes = len(id_dict)
    caltech_101_imgs = glob.glob("./datasets/caltech-101/101_ObjectCategories/*/*.jpg")
    caltech_101_imgs = [img_path.replace('\\', '/') for img_path in caltech_101_imgs]
    caltech_101_labels = [id_dict[img_path.split('/')[4]] for img_path in caltech_101_imgs]
    caltech_101_dataset = list(zip(caltech_101_imgs, caltech_101_labels))
    random.seed(0)
    random.shuffle(caltech_101_dataset)
    # 划分训练集和测试集，70%训练集，30%测试集
    train_size = int(0.7 * len(caltech_101_dataset))
    train_set, test_set = caltech_101_dataset[:train_size], caltech_101_dataset[train_size:]
    # 分离训练集的图片地址和标签
    train_imgs, train_labels = zip(*train_set)
    # 分离测试集的图片地址和标签
    test_imgs, test_labels = zip(*test_set)
    return list(train_imgs), list(train_labels), list(test_imgs), list(test_labels), num_classes


def read_data_food_101():
    id_dict = {}
    for i, line in enumerate(open('./datasets/food-101/food-101_taglist.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    num_classes = len(id_dict)
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    with open('./datasets/food-101/meta/train.txt', 'r') as f:
        for line in f:
            image = line.replace('\n', '') + '.jpg'
            label = line.split('/')[0]
            train_imgs.append(os.path.join('./datasets/food-101/images', image).replace('\\', '/'))
            train_labels.append(id_dict[label])
    with open('./datasets/food-101/meta/test.txt', 'r') as f:
        for line in f:
            image = line.replace('\n', '') + '.jpg'
            label = line.split('/')[0]
            test_imgs.append(os.path.join('./datasets/food-101/images', image).replace('\\', '/'))
            test_labels.append(id_dict[label])
    return train_imgs, train_labels, test_imgs, test_labels, num_classes


class CIFAR100_handler_train(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.transform = transform
        torch.manual_seed(1)
        np.random.seed(1)
        # print("self.YT", self.Y)
        for i in range(len(self.X)):
            r = random.randint(0, 99)
            if self.Y[i] == r:
                self.YT[i] = 1
                self.Y[i] = r
            else:
                self.YT[i] = 0
                self.Y[i] = r

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)


class DatasetHandlerTrain(Dataset):
    def __init__(self, X, Y, num_classes, transform):
        self.X = X
        self.Y = Y
        self.class_num = num_classes
        self.YT = torch.empty(len(self.Y))
        self.transform = transform
        torch.manual_seed(1)
        np.random.seed(1)
        # print("self.YT", self.Y)
        for i in range(len(self.X)):
            r = random.randint(0, num_classes - 1)
            if self.Y[i] == r:
                self.YT[i] = 1
                self.Y[i] = r
            else:
                self.YT[i] = 0
                self.Y[i] = r

    def __getitem__(self, index):
        x = Image.open(self.X[index])
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)


class CIFAR100HandlerTrainGenedClip(Dataset):
    def __init__(self, X, Y, transform=None):
        self.transform = transform
        self.X = X
        with open('./datasets/CIFAR100/train_label_tf.txt', "r") as file:
            lines = file.readlines()
            self.YT = [int(line) for line in lines]
        with open('./datasets/CIFAR100/train_label_r.txt', "r") as file:
            lines = file.readlines()
            self.Y = [int(line) for line in lines]
        # print("self.YT", self.YT)

    def __getitem__(self, index):
        # print(self.X[index].shape)
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        # x = Image.open(self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)


class DatasetHandlerTrainGenedClip(Dataset):
    def __init__(self, X, Y, dataset_name, transform):
        self.X = X
        self.dataset_name = dataset_name
        self.transform = transform
        with open(f'./datasets/{dataset_name}/train_label_tf.txt', "r") as file:
            lines = file.readlines()
            self.YT = [int(line) for line in lines]
        with open(f'./datasets/{dataset_name}/train_label_r.txt', "r") as file:
            lines = file.readlines()
            self.Y = [int(line) for line in lines]
        # print("self.YT", self.YT)

    def __getitem__(self, index):
        # print(self.X[index].shape)
        # x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = Image.open(self.X[index])
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)


# class CIFAR100_handler_train_clip(Dataset):
#     def __init__(self, X, Y, input_size, transform=None):
#         self.X = X
#         self.Y = Y
#         self.YT = torch.empty(len(self.Y))
#         self.transform = get_transform(input_size)
#         self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
#         torch.manual_seed(1)
#         np.random.seed(1)
#         clip_model = load_clip()
#         info = load_taglist(dataset="CIFAR100")
#         taglist_label = info["taglist"]
#         label_embed_label = build_clip_label_embedding(clip_model, taglist_label)
#         label_embed = label_embed_label.repeat(1, 1, 1)
#         label_embed = label_embed.to(device)
#         for i in range(len(self.X)):
#             print("i = ", i)
#             r = random.randint(0, 99)
#             x = Image.fromarray(np.uint8(self.X[i]).transpose((1, 2, 0)))
#             # x = Image.open(self.X[index]).convert('RGB')
#             imgs = self.transform(x).unsqueeze(0)
#             imgs = imgs.to(device)
#             # print("imgs", imgs.size())
#             image_embeds = clip_model.encode_image(imgs).unsqueeze(1)
#             image_embeds = image_embeds.to(device)
#             image_to_label = image_embeds.repeat(1, 100, 1)
#             output = self.cos(image_to_label, label_embed)
#             # print("torch.max(output,dim = 1)", torch.max(output, dim=1))
#             _, labels_g = torch.max(output, dim=1)
#             if labels_g == r:
#                 self.YT[i] = 1
#                 self.Y[i] = r
#                 file = open('./datasets/CIFAR100/train_label_tf.txt', 'a')
#                 file.write("1\n")
#                 file.close()
#             else:
#                 self.YT[i] = 0
#                 self.Y[i] = r
#                 file = open('./datasets/CIFAR100/train_label_tf.txt', 'a')
#                 file.write("0\n")
#                 file.close()
#             file = open('./datasets/CIFAR100/train_label_r.txt', 'a')
#             file.write(str(r) + '\n')
#             file.close()
#
#     def __getitem__(self, index):
#         # print(self.X[index].shape)
#         x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
#         # x = Image.open(self.X[index]).convert('RGB')
#         x = self.transform(x)
#         y = self.Y[index]
#         yt = self.YT[index]
#         return x, y, yt
#
#     def __len__(self):
#         return len(self.X)
#
#
# class DatasetHandlerTrainClip(Dataset):
#     def __init__(self, X, Y, dataset_name, num_classes, transform=get_transform()):
#         self.X = X
#         self.Y = Y
#         self.YT = torch.empty(len(self.Y))
#         self.transform = transform
#         self.dataset_name = dataset_name
#         self.class_num = num_classes
#         self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
#         torch.manual_seed(1)
#         np.random.seed(1)
#         clip_model = load_clip()
#         info = load_taglist(dataset=dataset_name)
#         taglist_label = info["taglist"]
#         label_embed_label = build_clip_label_embedding(clip_model, taglist_label)
#         label_embed = label_embed_label.repeat(1, 1, 1)
#         label_embed = label_embed.to(device)
#         for i in range(len(self.X)):
#             print("i = ", i)
#             r = random.randint(0, num_classes - 1)
#             # x = Image.fromarray(np.uint8(self.X[i]).transpose((1, 2, 0)))
#             x = Image.open(self.X[i])
#             imgs = self.transform(x).unsqueeze(0)
#             imgs = imgs.to(device)
#             # print("imgs", imgs.size())
#             image_embeds = clip_model.encode_image(imgs).unsqueeze(1)
#             image_embeds = image_embeds.to(device)
#             image_to_label = image_embeds.repeat(1, num_classes, 1)
#             output = self.cos(image_to_label, label_embed)
#             # print("torch.max(output,dim = 1)", torch.max(output, dim=1))
#             _, labels_g = torch.max(output, dim=1)
#             if labels_g == r:
#                 self.YT[i] = 1
#                 self.Y[i] = r
#                 file = open(f'./datasets/{dataset_name}/train_label_tf.txt', 'a')
#                 file.write("1\n")
#                 file.close()
#             else:
#                 self.YT[i] = 0
#                 self.Y[i] = r
#                 file = open(f'./datasets/{dataset_name}/train_label_tf.txt', 'a')
#                 file.write("0\n")
#                 file.close()
#             file = open(f'./datasets/{dataset_name}/train_label_r.txt', 'a')
#             file.write(str(r) + '\n')
#             file.close()
#
#     def __getitem__(self, index):
#         # print(self.X[index].shape)
#         # x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
#         x = Image.open(self.X[index])
#         x = self.transform(x)
#         y = self.Y[index]
#         yt = self.YT[index]
#         return x, y, yt
#
#     def __len__(self):
#         return len(self.X)


class CIFAR100_handler_test(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)

    def __getitem__(self, index):
        # print(self.X[index].shape)
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        # x = Image.open(self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class DatasetHandlerTest(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        # print(self.X[index].shape)
        # x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = Image.open(self.X[index])
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


#
# def data_gen_tf(train_set, train_label, test_set, test_label, input_size, batch_size, num_workers):
#     train_dataset = CIFAR100_handler_train_gened_clip(train_set, train_label, input_size)
#     train_loader = DataLoader(dataset=train_dataset,
#                               batch_size=batch_size,
#                               shuffle=True,
#                               num_workers=num_workers)
#     test_dataset = CIFAR100_handler_test(test_set, test_label, input_size)
#     test_loader = DataLoader(dataset=test_dataset,
#                              batch_size=batch_size,
#                              shuffle=True,
#                              num_workers=num_workers)
#     return train_loader, test_loader


def get_data_handler(dataset, pattern, input_size):
    if dataset == 'CIFAR100':
        train_data, train_label, test_data, test_label = read_data_cifar_100()
        if pattern == "train":
            datahandler = CIFAR100_handler_train(train_data, train_label,
                                                 transform=TransformTwice(get_transform(input_size)))
        elif pattern == "val":
            datahandler = CIFAR100_handler_test(test_data, test_label, input_size)
    else:
        if dataset == 'tiny-imagenet-200':
            train_data, train_label, test_data, test_label, num_classes = read_data_tiny_imagenet_200()
        elif dataset == 'EuroSAT':
            train_data, train_label, test_data, test_label, num_classes = read_data_eruosat()
        elif dataset == 'stanford_cars':
            train_data, train_label, test_data, test_label, num_classes = read_data_stanford_cars()
        elif dataset == 'caltech-101':
            train_data, train_label, test_data, test_label, num_classes = read_data_caltech_101()
        elif dataset == 'food-101':
            train_data, train_label, test_data, test_label, num_classes = read_data_food_101()

        if pattern == "train":
            datahandler = DatasetHandlerTrain(train_data, train_label, num_classes=num_classes,
                                              transform=TransformTwice(get_transform(input_size)))
        elif pattern == "val":
            datahandler = DatasetHandlerTest(test_data, test_label, get_transform(input_size))

    return datahandler


def get_data_handler_gened_clip(dataset, pattern, input_size):
    if dataset == 'CIFAR100':
        train_data, train_label, test_data, test_label = read_data_cifar_100()
        if pattern == "train":
            datahandler = CIFAR100HandlerTrainGenedClip(train_data, train_label,
                                                        transform=TransformTwice(get_transform(input_size)))
        elif pattern == "val":
            datahandler = CIFAR100_handler_test(test_data, test_label, input_size)
    else:
        if dataset == 'tiny-imagenet-200':
            train_data, train_label, test_data, test_label, num_classes = read_data_tiny_imagenet_200()
        elif dataset == 'EuroSAT':
            train_data, train_label, test_data, test_label, num_classes = read_data_eruosat()
        elif dataset == 'stanford_cars':
            train_data, train_label, test_data, test_label, num_classes = read_data_stanford_cars()
        elif dataset == 'caltech-101':
            train_data, train_label, test_data, test_label, num_classes = read_data_caltech_101()
        elif dataset == 'food-101':
            train_data, train_label, test_data, test_label, num_classes = read_data_food_101()

        if pattern == "train":
            datahandler = DatasetHandlerTrainGenedClip(train_data, train_label, dataset_name=dataset,
                                                       transform=TransformTwice(get_transform(input_size)))
        elif pattern == "val":
            datahandler = DatasetHandlerTest(test_data, test_label, get_transform(input_size))

    return datahandler


def load_datasets(
        dataset: str,
        pattern: str,
        img_size: int,
        batch_size: int,
        num_workers: int
) -> Tuple[DataLoader, Dict]:
    dataset_root = "./datasets/" + dataset
    tag_file = dataset_root + f"/{dataset}_taglist.txt"
    with open(tag_file, "r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]

    datahandler = get_data_handler(dataset, pattern, img_size)
    loader = DataLoader(dataset=datahandler, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    info = {
        "taglist": taglist
    }

    return loader, info


def load_datasets_gened_clip(
        dataset: str,
        pattern: str,
        img_size: int,
        batch_size: int,
        num_workers: int
) -> Tuple[DataLoader, Dict]:
    dataset_root = "./datasets/" + dataset
    tag_file = dataset_root + f"/{dataset}_taglist.txt"
    with open(tag_file, "r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]

    datahandler = get_data_handler_gened_clip(dataset, pattern, img_size)
    loader = DataLoader(dataset=datahandler, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    info = {
        "taglist": taglist
    }

    return loader, info


def divide_labeled_or_not(dataset, input_size):
    data_handler = get_data_handler(dataset, pattern='train', input_size=input_size)
    indices_yt_0 = torch.nonzero(torch.eq(data_handler.YT, 0)).squeeze().tolist()
    indices_yt_1 = torch.nonzero(torch.eq(data_handler.YT, 1)).squeeze().tolist()
    unlabeled_dataset = Subset(data_handler, indices_yt_0)
    labeled_dataset = Subset(data_handler, indices_yt_1)
    return labeled_dataset, unlabeled_dataset


if __name__ == '__main__':
    # train_data, train_label, test_data, test_label = read_data_cifar_100()

    # train_imgs, train_labels, test_imgs, test_labels, num_classes = read_data_tiny_imagenet_200()
    #
    # train_imgs_e, train_labels_e, test_imgs_e, test_labels_e, num_classes_e = read_data_eruosat()
    #
    # TinyImageNet_dataset = DatasetHandlerTrain(train_imgs, train_labels, 200, get_transform(224))

    # cifar100_dataset = CIFAR100_handler_train(train_data, train_label, 224)

    # id_dict = {}
    # for i, line in enumerate(open('./datasets/tiny-imagenet-200/wnids.txt', 'r')):
    #     id_dict[line.replace('\n', '')] = i
    # TinyImageNet_dataset = TinyImageNet_handler_train(root="./datasets/tiny-imagenet-200", Y=id_dict, input_size=224)

    # TinyImageNet_dataset = DatasetHandlerTrainClip(train_imgs, train_labels, dataset_name='tiny-imagenet-200',
    #                                                num_classes=200, transform=get_transform(224))

    # 使用一个 set 来记录包含 1 的类别
    all_categories = set()
    categories_with_1 = set()

    # 遍历 dataloader 中的数据
    for imgs, labels, labels_tf in loader:
        # 只考虑 labels_tf 等于 1 的情况
        all_categories.update(labels.tolist())
        categories_with_1.update(labels[labels_tf == 1].tolist())

    # 找出不包含 1 的类别
    categories_without_1 = all_categories - categories_with_1

    print("food-101中不包含True标签的类别:", categories_without_1)

    # print('_________________')
    # root_dir = './datasets/tiny-imagenet-200'
    # words_file = os.path.join(root_dir, "words.txt")
    # wnids_file = os.path.join(root_dir, "wnids.txt")
    # set_nids = set()
    # with open(wnids_file, 'r') as fo:
    #     data = fo.readlines()
    #     for entry in data:
    #         set_nids.add(entry.strip("\n"))
    #
    # class_to_label = {}
    # with open(words_file, 'r') as fo:
    #     data = fo.readlines()
    #     for entry in data:
    #         words = entry.split("\t")
    #         if words[0] in set_nids:
    #             class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]
    #
    # id_dict = {}
    # for i, line in enumerate(open('./datasets/tiny-imagenet-200/wnids.txt', 'r')):
    #     id_dict[line.replace('\n', '')] = i
    #
    # for key, value in id_dict.items():
    #     with open('./datasets/tiny-imagenet-200/tiny-imagenet-200_taglist.txt', 'a') as f:
    #         f.write(class_to_label[key] + "\n")
    # print('_________________')
    # root_dir = './datasets/caltech-101/101_ObjectCategories'
    # classes = glob.glob('./datasets/caltech-101/101_ObjectCategories/*')
    # for path in classes:
    #     classes_name = os.path.basename(path)
    #     with open('./datasets/caltech-101/caltech-101_taglist.txt', 'a') as f:
    #         f.write(classes_name + "\n")
