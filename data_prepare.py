from torchvision import transforms, datasets
import os
import torch
from PIL import Image
import scipy.io as scio
import argparse

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def ImageNetData(args):
# data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {}
    #image_datasets['train'] = datasets.ImageFolder(os.path.join(args.data_dir, 'ILSVRC2012_img_train'), data_transforms['train'])

    image_datasets['train'] = ImageNetTrainDataSet(os.path.join(args.data_dir, 'list_train_3600.txt'),
                                           os.path.join(args.data_dir,'photos_new'),
                                           data_transforms['train'])
    '''for i,(input, labels) in enumerate(image_datasets['train']):
        print(labels)
        print("------------")'''
    image_datasets['val'] = ImageNetValDataSet(os.path.join(args.data_dir, 'list_val_3600.txt'),
                                               os.path.join(args.data_dir, 'photos_new'),
                                               data_transforms['val'])

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers) for x in ['train', 'val']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes

class ImageNetTrainDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, root_d, data_transforms):
        '''label_array = scio.loadmat(img_label)['synsets']
        label_dic = {}
        for i in  range(1000):
            label_dic[label_array[i][0][1][0]] = i'''

        self.data_transforms = data_transforms
        self.root_dir = root_dir
        self.root_d = root_d
        self.img_path, self.label_dic = self.read_from_train_txt()
        self.imgs = self._make_dataset()
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        data, label = self.imgs[item]
        img = Image.open(data).convert('RGB')
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return img, label

    #read from train txt     line like: Three/IMG_2608.jpg   15
    def read_from_train_txt(self):
        label_dict = {}
        img_path = []
        with open(self.root_dir, "r") as train_txt:
            lines = train_txt.readlines();
        for line in lines:
            path_label = line.split(' ')
            path = path_label[0]
            label = path_label[1].split('\n')[0]
            #print(label)
            #print("----------")	   
            img_path.append(os.path.join(self.root_d, path))
            try:
                label_dict[os.path.join(self.root_d, path)] = int(label)
            except:
                print(label)
        return img_path, label_dict

    def _make_dataset(self):
        #class_to_idx = self.label_dic
        images = []
        '''
	dir = os.path.expanduser(self.root_dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self._is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)   '''
        images =  list(self.label_dic.items())
        print(len(images))   
        '''for i, (inputs, labels) in enumerate(images):
            print(labels)
            print("--------------")'''
        return images

    def _is_image_file(self, filename):
        """Checks if a file is an image.
        Arg           filename (string): path to a file
        Returns:
            bool: True if the filename ends with a known image extension
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

class ImageNetValDataSet(torch.utils.data.Dataset):
    def __init__(self, img_path, img_label, data_transforms):
        self.data_transforms = data_transforms
        #img_names = os.listdir(img_path)
        #img_names.sort()
        #self.img_path = [os.path.join(img_path, img_name) for img_name in img_names]
        with open(img_path,"r") as input_file:
            lines = input_file.readlines()
            self.img_label = [(line.split('/')[0]) for line in lines]
            self.img_path = [os.path.join(img_label, line.split('\n')[0]) for line in lines]
    
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img = Image.open(self.img_path[item]).convert('RGB')
        label = self.img_label[item]
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return img, label
