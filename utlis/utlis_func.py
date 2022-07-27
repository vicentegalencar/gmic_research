import os
import optparse as op
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
from collections import Counter
from utlis.crop_breast import suppress_artifacts, crop_max_bg
import cv2
import random
import pandas as pd

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from utlis.mEfficientNet import EfficientNet
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import wandb
train_h, train_w = 1536, 768
# train_h, train_w = 2944, 1920


def config_dataset(params, config, csv_path):
    config.model_name = params.model_name  # String. Network model
    config.use_pretrained = params.use_pretrained # Boolean. Use pretrained network
    config.use_augmentation = params.use_augmentation # Boolean. Use data augmentation
    config.use_gamma = params.use_gamma # Boolean. Use gamma correction
    config.use_crop = params.use_crop # Boolean. Use image cropping
    config.num_workers = params.num_workers # number of CPU threads
    config.batch_size = params.batch_size  # input batch size for training (default: 64)
    config.epochs = params.epochs  # number of epochs to train (default: 10)
    config.lr = params.lr # learning rate (default: 0.01)
    config.seed = params.seed  # random seed (default: 42) #103 #414
    config.earlystop_patience = params.earlystop_patience  # epochs to wait without improvements before stopping training
    config.reduceLR_patience = params.reduceLR_patience  # epoch to wait without improvements before reducing learning rate
    config.fc_size = params.fc_size # size of fully connected layer
    config.manufacturer_train = params.manufacturer_train
    kwargs = {'num_workers': config.num_workers, 'pin_memory': True}
    cv2.setNumThreads(0)
    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(config.seed)  # python random seed
    torch.manual_seed(config.seed)  # pytorch random seed
    np.random.seed(config.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    if config.use_augmentation:
        if config.use_gamma:
            trans_train = transforms.Compose([
                transforms.Resize(size=[train_h, train_w]),
                GammaCorrection(),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5], std=[0.5])
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            trans_train = transforms.Compose([
                transforms.Resize(size=[train_h, train_w]),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5], std=[0.5])
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        if config.use_gamma:
            trans_train = transforms.Compose([
                transforms.Resize(size=[train_h, train_w]),
                GammaCorrection(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            trans_train = transforms.Compose([
                transforms.Resize(size=[train_h, train_w]),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5], std=[0.5])
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    if config.use_gamma:
        trans_val_test = transforms.Compose([
            transforms.Resize(size=[train_h, train_w]),
            GammaCorrection(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        trans_val_test = transforms.Compose([
            transforms.Resize(size=[train_h, train_w]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load csv file
    df_full = pd.read_csv(csv_path["df_filepath"], index_col='ImageId')

    var1 = config.manufacturer_train

    manufacturers = ['FUJIFILM Corporation',
                     'SIEMENS',
                     'HOLOGIC, Inc.',
                     'Philips Digital Mammography Sweden AB',
                     'KONICA MINOLTA',
                     'Sectra Imtec AB'
                    ]
     # cancer df
    df_cancer = df_full.loc[(df_full['ImageOutcome'] == 1) &
                             ((df_full['ImageViewPosition'] == 'L-MLO') | (df_full['ImageViewPosition'] == 'R-MLO') |
                              (df_full['ImageViewPosition'] == 'L-CC') | (df_full['ImageViewPosition'] == 'R-CC')) &
                            #  (df_full['AxSubOutcome'] != 2) &
                             (df_full['ImageManufacturer'] == 'SIEMENS') &
                             (df_full['ImageModality'] != 'OT')].sample(1814, random_state=config.seed)

    df_cancer2 = df_full.loc[(df_full['ImageOutcome'] == 1) &
                             ((df_full['ImageViewPosition'] == 'L-MLO') | (df_full['ImageViewPosition'] == 'R-MLO') |
                              (df_full['ImageViewPosition'] == 'L-CC') | (df_full['ImageViewPosition'] == 'R-CC')) &
                            #  (df_full['AxSubOutcome'] != 2) &
                             (df_full['ImageManufacturer'] == 'HOLOGIC, Inc.') &
                             (df_full['ImageModality'] != 'OT')].sample(1172, random_state=config.seed)
    
    df_cancer3 = df_full.loc[(df_full['ImageOutcome'] == 1) &
                             ((df_full['ImageViewPosition'] == 'L-MLO') | (df_full['ImageViewPosition'] == 'R-MLO') |
                              (df_full['ImageViewPosition'] == 'L-CC') | (df_full['ImageViewPosition'] == 'R-CC')) &
                            #  (df_full['AxSubOutcome'] != 2) &
                             (df_full['ImageManufacturer'] == 'FUJIFILM Corporation') &
                             (df_full['ImageModality'] != 'OT')].sample(1814, random_state=config.seed)

    df_cancer = df_cancer.append(df_cancer2)
    df_cancer = df_cancer.append(df_cancer3)
    # no cancer df
    df_control = df_full.loc[(df_full['ImageOutcome'] == 0) &
                             ((df_full['ImageViewPosition'] == 'L-MLO') | (df_full['ImageViewPosition'] == 'R-MLO') |
                              (df_full['ImageViewPosition'] == 'L-CC') | (df_full['ImageViewPosition'] == 'R-CC')) &
                            #  (df_full['AxSubOutcome'] != 2) &
                             (df_full['ImageManufacturer'] == var1) &
                             (df_full['ImageModality'] != 'OT')].sample(2098, random_state=config.seed)

    df_control1 = df_full.loc[(df_full['ImageOutcome'] == 0) &
                             ((df_full['ImageViewPosition'] == 'L-MLO') | (df_full['ImageViewPosition'] == 'R-MLO') |
                              (df_full['ImageViewPosition'] == 'L-CC') | (df_full['ImageViewPosition'] == 'R-CC')) &
                            #  (df_full['AxSubOutcome'] != 2) &
                             (df_full['ImageManufacturer'] == 'HOLOGIC, Inc.') &
                             (df_full['ImageModality'] != 'OT')].sample(2098, random_state=config.seed)
    
    df_control2 = df_full.loc[(df_full['ImageOutcome'] == 0) &
                             ((df_full['ImageViewPosition'] == 'L-MLO') | (df_full['ImageViewPosition'] == 'R-MLO') |
                              (df_full['ImageViewPosition'] == 'L-CC') | (df_full['ImageViewPosition'] == 'R-CC')) &
                            #  (df_full['AxSubOutcome'] != 2) &
                             (df_full['ImageManufacturer'] == 'FUJIFILM Corporation') &
                             (df_full['ImageModality'] != 'OT')].sample(2098, random_state=config.seed)
    df_control = df_control.append(df_control1)
    df_control = df_control.append(df_control2)


    df = df_cancer.append(df_control)
    # Create the index splits for training, validation and test using the ClientId information
    # train_size = 0.8
    train_size = 0.82
    # train_size = 0.77543169772
    num_train = len(df.ClientId.unique())
    split = int(np.floor(train_size * num_train))
    split2 = int(np.floor((1. - train_size) / 2. * num_train))

    g = df.groupby('ClientId')
    df_train = df[g.ngroup().isin(np.random.choice(g.ngroups, split, replace=False))]
    df_test = df.drop(df_train.index, inplace=False)
    g = df_test.groupby('ClientId')
    df_val = df_test[g.ngroup().isin(np.random.choice(g.ngroups, split2, replace=False))]
    df_test.drop(df_val.index, inplace=True)

    df['TrainCode'] = ''
    df.loc[df_train.index, 'TrainCode'] = 0
    df.loc[df_val.index, 'TrainCode'] = 1
    df.loc[df_test.index, 'TrainCode'] = 2
    
    traindataset = ImageDataset(csv_file=df_train, root_dir=csv_path["img_folderpath"], transform=trans_train, args=params)
    valdataset = ImageDataset(csv_file=df_val, root_dir=csv_path["img_folderpath"], transform=trans_val_test, args=params)
    testdataset = ImageDataset(csv_file=df_test, root_dir=csv_path["img_folderpath"], transform=trans_val_test, args=params)

    trainloader = DataLoader(traindataset, batch_size=config.batch_size, shuffle=True, drop_last=True, **kwargs)#, sampler=sampler
    valloader = DataLoader(valdataset, batch_size=config.batch_size, shuffle=True, drop_last=True, **kwargs)
    testloader = DataLoader(testdataset, batch_size=config.batch_size, shuffle=False, drop_last=False, **kwargs)
    print('Number of images with cancer: {}'.format(len(df_cancer)))
    print('Number of images without cancer: {}'.format(len(df_control)))
    print('Total number of images: {}'.format(len(df)))
    print('Number of train images: {}'.format(len(df_train)))
    print('Number of vald images: {}'.format(len(df_val)))
    print('Number of test images: {}'.format(len(df_test)))
    print('')

    return config, df, df_train, df_val, df_test, trainloader, valloader, testloader


def optionFlags():
    parser = op.OptionParser("%program [options]")# headerfile datafile dataXMLfile")
    anGroup = op.OptionGroup(parser, 'anParams')

    anGroup.add_option("-p", "--use_pretrained",
                       action="store_true",
                       help="Switch on the use of pretrained network (default is False)",
                       dest="use_pretrained", default=False)
    
    anGroup.add_option("-a", "--use_augmentation",
                       action="store_true",
                       help="Switch on the use of data augmentation (default is False)",
                       dest="use_augmentation", default=False)
    
    anGroup.add_option("-c", "--use_gamma",
                       action="store_true",
                       help="Switch on the use of gamma correction (default is False)",
                       dest="use_gamma", default=False)
    
    anGroup.add_option("-k", "--use_crop",
                       action="store_true",
                       help="Switch on the use of image cropping (default is False)",
                       dest="use_crop", default=False)

    parser.add_option("-m", "--model_name",
                      help="Defines the model to use (default is ResNet-50)",
                      action="store", type="string", dest="model_name", default='ResNet-50')

    parser.add_option("-z", "--manufacturer_train",
                      help="Defines the manufacturer to train on (default is SIEMENS)",
                      action="store", type="string", dest="manufacturer_train", default='SIEMENS')

    parser.add_option("-g", "--gpu_device",
                      help="Defines the GPU device to use (default is 0)",
                      type="int", dest="gpu_device", default=0)

    parser.add_option("-d", "--seed",
                      help="Defines the seed (default is 42)",
                      type="int", dest="seed", default=42)

    parser.add_option("-w", "--num_workers",
                      help="Defines the number of workers for multithreading (default is 0)",
                      type="int", dest="num_workers", default=0)

    parser.add_option("-b", "--batch_size",
                      help="Defines the batch size (default is 24)",
                      type="int", dest="batch_size", default=24)

    parser.add_option("-e", "--epochs",
                      help="Defines the maximum number of epochs (default is 80)",
                      type="int", dest="epochs", default=80)

    parser.add_option("-s", "--earlystop_patience",
                      help="Defines the early stop patience (default is 15)",
                      type="int", dest="earlystop_patience", default=15)

    parser.add_option("-r", "--reduceLR_patience",
                      help="Defines the learning rate reduction patience (default is 5)",
                      type="int", dest="reduceLR_patience", default=5)

    parser.add_option("-f", "--fc_size",
                      help="Defines the size of the fully connected layer (default is 1024)",
                      type="int", dest="fc_size", default=1024)

    parser.add_option("-l", "--lr",
                      help="Defines the initial learning rate (default is 0.005)",
                      type="float", dest="lr", default=0.01)

    parser.add_option_group(anGroup)
    (opts, args) = parser.parse_args()

    # assert len(args) == 0
    return opts


def Net(model_name='ResNet-18', use_pretrained=True, fc_size=1024, freeze=False):
    print('Loading pretrained model : {} ...'.format(model_name))
    if model_name == 'ResNet-18':
        net = models.resnet18(pretrained=use_pretrained, progress=True)
    elif model_name == 'ResNet-34':
        net = models.resnet34(pretrained=use_pretrained, progress=True)
    elif model_name == 'ResNet-50':
        net = models.resnet50(pretrained=use_pretrained, progress=True)
    elif model_name == 'ResNet-101':
        net = models.resnet101(pretrained=use_pretrained, progress=True)
    elif model_name == 'ResNet-152':
        net = models.resnet152(pretrained=use_pretrained, progress=True)
    elif model_name == 'AlexNet':
        net = models.alexnet(pretrained=use_pretrained)
    elif model_name == 'SqueezeNet':
        net = models.squeezenet1_0(pretrained=use_pretrained)
    elif model_name == 'VGG16':
        net = models.vgg16(pretrained=use_pretrained)
    elif model_name == 'DenseNet':
        net = models.densenet161(pretrained=use_pretrained)
    elif model_name == 'Inception':
        net = models.inception_v3(pretrained=use_pretrained)
    elif model_name == 'GoogleNet':
        net = models.googlenet(pretrained=use_pretrained)
    elif model_name == 'ShuffleNet':
        net = models.shufflenet_v2_x1_0(pretrained=use_pretrained)
    elif model_name == 'MobileNet':
        net = models.mobilenet_v2(pretrained=use_pretrained)
    elif model_name == 'ResNext-50':
        net = models.resnext50_32x4d(pretrained=use_pretrained)
    elif model_name == 'MNASNet':
        net = models.mnasnet1_0(pretrained=use_pretrained)
    elif 'EfficientNet' in model_name:
        if use_pretrained:
            net = EfficientNet.from_pretrained(model_name.lower())
        else:
            net = EfficientNet.from_name(model_name.lower())
    else:
        print('ERROR: model {} was not recognised ...')
        exit()
        
    for name in list(net.named_modules()):
        if 'fc' in name[0]:
            classifier_name = name[0]
        elif 'classifier' in name[0]:
            classifier_name = name[0]    
    
    if use_pretrained and freeze:
        print('Freezing learning on initial layers ...')
        for name, param in net.named_parameters():
            if ("bn" not in name):
                param.requires_grad = False
                # print("NAME \t", name)

        # ct = 0
        # for name, child in net.named_children():
        #     ct += 1
        #     if ct < 9: # 9 makes it train only the fully connected layer
        #         print('NAMED CHILDREN \t', name)
        #         for name2, params in child.named_parameters():
        #             if ("bn" not in name2):
        #                 params.requires_grad = False
        #                 print("\t\t NAMED PARAMETER \t", name2)

    # print(net)
    print('Customising last fully connected layer ...')
    print('Classifier module name : {}'.format(classifier_name))
    if classifier_name == 'classifier':
        net.classifier = nn.Sequential(nn.Linear(net.classifier.in_features, fc_size), nn.ReLU(), nn.Dropout(0.5), nn.Linear(fc_size, 1))
    elif classifier_name == '_fc':
        net._fc = nn.Sequential(nn.Linear(net._fc.in_features, fc_size), nn.ReLU(), nn.Dropout(0.5), nn.Linear(fc_size, 1))
    else:
        net.fc = nn.Sequential(nn.Linear(net.fc.in_features, fc_size), nn.ReLU(), nn.Dropout(0.5), nn.Linear(fc_size, 1))
    return net
    
    # if 'EfficientNet' in model_name:
    #     net._fc = nn.Sequential(nn.Linear(net._fc.in_features, fc_size), nn.ReLU(), nn.Dropout(0.5), nn.Linear(fc_size, 1))
    # elif model_name == 'AlexNet':
    #     net.classifier = nn.Sequential(nn.Linear(net.classifier[1].in_features, fc_size), nn.ReLU(), nn.Dropout(0.5), nn.Linear(fc_size, 1))
    # elif model_name == 'SqueezeNet':
    #     net.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))
    # elif model_name == 'VGG16':
    #     net.classifier = nn.Sequential(nn.Linear(net.classifier[0].in_features, fc_size), nn.ReLU(), nn.Dropout(0.5), nn.Linear(fc_size, 1))
    # elif model_name == 'DenseNet':
    #     net.classifier = nn.Sequential(nn.Linear(net.classifier.in_features, fc_size), nn.ReLU(), nn.Dropout(0.5), nn.Linear(fc_size, 1))
    # else:
    #     net.fc = nn.Sequential(nn.Linear(net.fc.in_features, fc_size), nn.ReLU(), nn.Dropout(0.5),nn.Linear(fc_size, 1))
    # return net


def binary_acc(y_pred, y_test):
    # y_pred_tag = torch.round(torch.sigmoid(y_pred))
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


def imgshow_tensor(input, nrow=1, name="img", val_range=(-2.1179, 2.6400)):
    val_range = (-2.1179, 2.6400)
    grid_image = torchvision.utils.make_grid(input, normalize=False, nrow=nrow, padding=0)
    np_grid_image = grid_image.cpu().detach().numpy()
    img_tmp = np.transpose(np_grid_image, (1, 2, 0)).squeeze().astype(np.float32)
    plt.axis("off")
    plt.imshow(img_tmp)
    # plt.savefig(os.path.join("./", name), bbox_inches='tight')
    plt.show()
    # return img_tmp


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        # if torch.isnan(metrics):
        #     return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


class GammaCorrection(object):    
    def __call__(self, image):
        data = Counter(np.ravel(image))
        if data.most_common()[1][0] > 85:
            imgTmp = TF.adjust_gamma(image, 4.0, gain=4)
        elif data.most_common()[1][0] > 70:
            imgTmp = TF.adjust_gamma(image, 3.0, gain=3)
        elif data.most_common()[1][0] > 35:
            imgTmp = TF.adjust_gamma(image, 2.0, gain=2)
        else:
            imgTmp = image
        return imgTmp


class ImageDataset(Dataset):
    """Image and label dataset."""

    def __init__(self, csv_file, root_dir, transform=None, args=None):
        """
        Args:
            csv_file (string): Csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.args = args

    def SquarePad(self, image, idx, ratio_hw):
        w, h = image.size
        place = (0, 0)
        if h / w > ratio_hw:
            if self.df.iloc[idx].ImageLaterality == 'R':
                place = (int(h / ratio_hw) - w, 0)
            result = Image.new(image.mode, (int(h // ratio_hw), h), 0)
        elif h / w == ratio_hw:
            return image
        else:
            result = Image.new(image.mode, (w, int(w * ratio_hw)), 0)
        result.paste(image, place)
        return result

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # opens png image
        img_path = os.path.join(self.root_dir, self.df.iloc[idx].ImageFilePath)
        imgTmp = Image.open(img_path)
        table = [i / 256 for i in range(65536)]
        imgTmp = imgTmp.point(table, 'L')
        data = None
        org_h = 0
        org_w = 0

        # if self.args.use_crop:
        img = np.asarray(imgTmp)
        org_h, org_w = img.shape
        org = img.copy()
        # if self.df.iloc[idx].ImageManufacturer == "FUJIFILM Corporation":
        pad=25
        # y
        org[0:pad, :] = 0
        org[(org.shape[0]-pad):org.shape[0], :] = 0
        # x
        # org[:, 0:pad] = 0
        # org[:, (org.shape[1]-pad):org.shape[1]] = 0

        (img_suppr, breast_mask) = suppress_artifacts(org, global_threshold=.1, fill_holes=True, smooth_boundary=True, kernel_size=15)
        img_breast_only, (x, y, w, h) = crop_max_bg(img_suppr, breast_mask)
        imgTmp = Image.fromarray(img_breast_only)

        # img = imgTmp.convert('RGB')
        img = imgTmp.convert('L')

        # Maybe not change 1536/768 -> org_h/org_w ?
        img = self.SquarePad(img, idx, 1536/768)

        viewpos = self.df.iloc[idx].ImageViewPosition[0]
        normalization = []

        if self.transform:
            data = self.transform(img)
            img_mean = torch.mean(data)
            img_std = torch.maximum(torch.std(data), torch.tensor(10**(-5)))
            normalization.append(transforms.Normalize(mean=[img_mean], std=[img_std]))
            if viewpos == 'R':
                flip = transforms.RandomHorizontalFlip(p=1)
                normalization.append(flip)
            elif viewpos == 'L':
                do_sth = "nothing"
            else:
                raise("Incorrect view position: {}".format(viewpos))
        transform_2 = transforms.Compose(normalization)
        data = transform_2(data)
        target = float(self.df.iloc[idx].ImageOutcome)
        target = torch.tensor(target)
        return data, target
