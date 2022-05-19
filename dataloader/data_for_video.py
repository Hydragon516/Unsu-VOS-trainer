import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import config as config

def cv_random_flip(img, label, flow):
    flip_flag = random.randint(0, 1)

    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        flow = flow.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label, flow

def randomCrop(image, label, flow):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)

    return image.crop(random_region), label.crop(random_region), flow.crop(random_region)

def randomRotation(image,label,flow):
    mode = Image.BICUBIC

    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        flow = flow.rotate(random_angle, mode)
    
    return image,label,flow

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5,15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0,20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0,30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    
    return image

def randomGaussian(image, mean=0.1, sigma=0.35):
    
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        
        return im
    
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])

    return Image.fromarray(np.uint8(img))

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0]-1)  
        randY = random.randint(0, img.shape[1]-1)  

        if random.randint(0,1) == 0:  
            img[randX, randY] = 0  
        else:  
            img[randX, randY] = 255 

    return Image.fromarray(img)  


class SalObjDataset(data.Dataset):
    def __init__(self):
        self.trainsize = config.TRAIN['img_size']
        main_image_root = os.path.join(config.DATA['data_root'], config.DATA['DAVIS_train_main'], 'RGB') + '/'
        main_gt_root = os.path.join(config.DATA['data_root'], config.DATA['DAVIS_train_main'], 'GT') + '/'
        main_flow_root = os.path.join(config.DATA['data_root'], config.DATA['DAVIS_train_main'], 'FLOW') + '/'

        self.main_images = [main_image_root + f for f in os.listdir(main_image_root) if f.endswith('.jpg')]
        self.main_gts = [main_gt_root + f for f in os.listdir(main_gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.main_flows = [main_flow_root + f for f in os.listdir(main_flow_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        self.main_images = sorted(self.main_images)
        self.main_gts = sorted(self.main_gts)
        self.main_flows = sorted(self.main_flows)

        self.sub_images = []
        self.sub_gts = []
        self.sub_flows = []

        if config.DATA['DAVIS_train_sub'] is not None:
            sub_image_root = os.path.join(config.DATA['data_root'], config.DATA['DAVIS_train_sub'], 'RGB') + '/'
            sub_gt_root = os.path.join(config.DATA['data_root'], config.DATA['DAVIS_train_sub'], 'GT') + '/'
            sub_flow_root = os.path.join(config.DATA['data_root'], config.DATA['DAVIS_train_sub'], 'FLOW') + '/'

            self.sub_images = [sub_image_root + f for f in os.listdir(sub_image_root) if f.endswith('.jpg')]
            self.sub_gts = [sub_gt_root + f for f in os.listdir(sub_gt_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.sub_flows = [sub_flow_root + f for f in os.listdir(sub_flow_root) if f.endswith('.jpg') or f.endswith('.png')]

            self.sub_images = sorted(self.sub_images)
            self.sub_gts = sorted(self.sub_gts)
            self.sub_flows = sorted(self.sub_flows)

            sub = list(zip(self.sub_images, self.sub_gts, self.sub_flows))
            random.shuffle(sub)
            self.sub_images, self.sub_gts, self.sub_flows = zip(*sub)

            self.sub_images = self.sub_images[:len(self.main_images)]
            self.sub_gts = self.sub_gts[:len(self.main_images)]
            self.sub_flows = self.sub_flows[:len(self.main_images)]

        self.total_images = self.main_images + list(self.sub_images)
        self.total_gts = self.main_gts + list(self.sub_gts)
        self.total_flows = self.main_flows + list(self.sub_flows)
        
        self.size = len(self.total_images)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        self.flows_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.total_images[index])
        gt = self.binary_loader(self.total_gts[index])
        flow = self.rgb_loader(self.total_flows[index])
        
        image, gt, flow = cv_random_flip(image, gt, flow)
        image, gt, flow = randomCrop(image, gt, flow)
        image, gt, flow = randomRotation(image, gt, flow)
        
        image = colorEnhance(image)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        flow = self.flows_transform(flow)
        
        return image, gt, flow

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('L')

    def resize(self, img, gt, flow):
        assert img.size == gt.size and gt.size == flow.size
        
        w, h = img.size
        
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), flow.resize((w, h), Image.NEAREST)
        else:
            return img, gt, flow

    def __len__(self):
        return self.size


def get_loader(shuffle=True, num_workers=12, pin_memory=False):

    dataset = SalObjDataset()
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.TRAIN['batch_size'],
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    return data_loader
    
class SalObjDataset_test(data.Dataset):
    def __init__(self, val):
        self.testsize = config.TRAIN['img_size']
        
        self.images = []
        self.gts = []
        self.flows = []

        folder = os.path.join(config.DATA['data_root'], val)
        valid_list = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
        
        for valid_name in valid_list:
            image_root = os.path.join(config.DATA['data_root'], val, valid_name, "RGB") + "/"
            gt_root = os.path.join(config.DATA['data_root'], val, valid_name, "GT") + "/"
            flow_root = os.path.join(config.DATA['data_root'], val, valid_name, "FLOW") + "/"

            new_images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
            new_gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
            new_flows = [flow_root + f for f in os.listdir(flow_root) if f.endswith('.jpg') or f.endswith('.png')]

            new_images = sorted(new_images)
            new_gts = sorted(new_gts)
            new_flows = sorted(new_flows)

            for i in range(len(new_flows)):
                self.images.append(new_images[i])
                self.gts.append(new_gts[i])
                self.flows.append(new_flows[i])

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.flows = sorted(self.flows)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.flows_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)), 
            transforms.ToTensor()])
        
        self.size = len(self.images)
    
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        flow = self.rgb_loader(self.flows[index])

        image = self.transform(image)
        flow = self.flows_transform(flow)
        
        name = self.images[index].split('/')[-1]
        valid_name = self.images[index].split('/')[-3]
        
        image_for_post = self.rgb_loader(self.images[index])
        image_for_post = image_for_post.resize((self.testsize, self.testsize))
        
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        info = [gt.size, valid_name, name]
        
        gt = self.gt_transform(gt)
        
        return image, gt, flow, info, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('L')
    
    def __len__(self):
        return self.size

def get_testloader(val, shuffle=False, num_workers=12, pin_memory=False):
    dataset = SalObjDataset_test(val)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.TRAIN['batch_size'],
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    return data_loader