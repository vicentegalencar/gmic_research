import os, sys, argparse, csv
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.modeling import gmic
from src.utilities.metric import compute_metric
#from src.data_loading.datasets import get_dataloader
from src.data_loading.datasets import get_dataloaderCBIS
from src.scripts.one_epoch import running_one_epoch, save_epoch

def train_net(net, output_path, epochs, image_size, batch_size, optimizer, lr_scheduler, datapath, device, threshold, writer, \
    beta, lr_rate, max_value, aug, num_chan):
    # dataloader
    #train_loader = get_dataloaderCBIS(datapath, 'train', image_size=image_size, batch_size=batch_size, shuffle=True, max_value=max_value, aug=aug, num_chan=num_chan)
    train_loader = get_dataloaderCBIS(datapath, 'train_rnd', image_size=image_size, batch_size=batch_size, shuffle=True, max_value=max_value, aug=aug, num_chan=num_chan)
    #val_loader = get_dataloaderCBIS(datapath, 'validate', image_size=image_size, batch_size=10, shuffle=False, max_value=max_value, aug=aug)
    # val_loader = get_dataloaderCBIS(datapath, 'test', image_size=image_size, batch_size=10, shuffle=False, max_value=max_value, aug=False, num_chan=num_chan)
    val_loader = get_dataloaderCBIS(datapath, 'test_rnd', image_size=image_size, batch_size=10, shuffle=False, max_value=max_value, aug=aug, num_chan=num_chan)
    #test_loader = get_dataloaderCBIS(datapath, 'test', image_size=image_size, batch_size=10, shuffle=False, max_value=max_value, aug=False, num_chan=num_chan)
    test_loader = get_dataloaderCBIS(datapath, 'test_rnd', image_size=image_size, batch_size=10, shuffle=False, max_value=max_value, aug=aug, num_chan=num_chan)

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    criterion = torch.nn.BCELoss().cuda()
    for epoch in range(epochs):
        lr = running_one_epoch(epoch, output_path, net, train_loader, val_loader, criterion, optimizer, device, threshold, beta, train_loss, train_acc, \
        valid_loss, valid_acc, writer)
        if lr > float(lr_rate)/10:
            lr_scheduler.step()
    print('#######################')
    print('==> training end, save models, test and save results')
    save_epoch(epoch, output_path, net, 'last_model.pth')
    #test_net(net, output_path, test_loader, device, threshold)
    test_net(net, test_loader, device, threshold)
    return train_loss, train_acc, valid_loss, valid_acc

@torch.no_grad()
def test_net(model, loader, device, threshold):
    # load val best to test
    model_path = os.path.join(output_path, 'val_best_model.pth')
    model.load_state_dict(torch.load(model_path)['model_state_dict'], strict=True)
    model.eval()

    prediction = np.empty(shape=[0, 2], dtype=np.int)
    for step, (imgs, labels, _) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        y_global, y_local, y_fusion, saliency_map = model(imgs)
        
        y_fusion = y_fusion[:,1:]
        
        result = np.concatenate([y_fusion.cpu().data.numpy(), labels.cpu().data.numpy()], axis=1)
        prediction = np.concatenate([prediction, result], axis=0)

    print('==> ### test metric ###')
    total = len(loader.dataset)
    TP, FN, TN, FP, acc, roc_auc = compute_metric(prediction, threshold)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    print('Total: %d'%(total))
    print('threshold: %.2f --- TP: %d --- FN: %d --- TN: %d --- FP: %d'%(threshold, TP, FN, TN, FP))
    print('acc: %f --- roc_auc: %f --- sensitivity: %f --- specificity: %f'%(acc, roc_auc, sensitivity, specificity))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # ap.add_argument("--model_path", type=str, default='models/sample_model_1.p',
    #     help="pretrained model path")
    ap.add_argument("--save_model_path", type=str, default='checkpoints/mymodel_1.pth',
        help="path for saving trained model")
    # ap.add_argument("--train_data_csv", type=str, default='mammo/train.csv', 
    #     help="path to csv file recording the training samples")
    # ap.add_argument("--val_data_csv", type=str, default='mammo/validate.csv', 
    #     help="path to csv file recording the validation samples")
    # ap.add_argument("--test_data_csv", type=str, default='mammo/test.csv', 
    #     help="path to csv file recording the test samples")
    ap.add_argument("--data_path", type=str, default='../../datasets/preprocessed/', 
        help="path to training and test data")
    ap.add_argument("--epochs", type=int, default=50,
        help="epoch numbers")
    ap.add_argument("--lr", type=float, default=4.134478662168656e-04,
        help="base learning rate")
    ap.add_argument("--lr_step", type=int, default=10,
        help="steps for decreasing learning rate")
    ap.add_argument("--bs", type=int, default=6,
        help="batch size")
    ap.add_argument("--is", type=int, default=(2944, 1920),
        help="image size")
    ap.add_argument("--beta", type=float, default=3.259162430057801e-06,
        help="beta")
    ap.add_argument("--percent_t", type=float, default=0.02,
        help="percent for top-T pooling")
    ap.add_argument("--augmentation", type=bool, default=False,
        help="whether to perform data augmentation during training")
    ap.add_argument("--v1_global", type=bool, default=False,
        help="use RN18 as v1_global")
    ap.add_argument("--gpuid", type=int, default=0,
        help="gpu id")
    ap.add_argument("--pretrain_global", type=bool, default=False,
        help="whether to perform pretrain from imagenet")
    ap.add_argument("--pretrain_local", type=bool, default=False,
        help="whether to perform pretrain from imagenet")
    ap.add_argument("--num_chan", type=int, default=3,
        help="batch size")
    
    
    args = vars(ap.parse_args())
    num_works = 4
    

    #pretrained_state_dict = args['model_path']
    output_path = args['save_model_path']

    lr_rate = args['lr']
    beta = args['beta']
    percent_t = args['percent_t']

    epochs = args['epochs']
    lr_step = args['lr_step']
    batch_size = args['bs']
    img_size = args['is']
    aug = args['augmentation']
    use_v1_global = args['v1_global']
    pretrain_local = args['pretrain_local']
    pretrain_global = args['pretrain_global']
    
    # Train_data = args['train_data_csv']
    # Val_data = args['val_data_csv']
    # Test_data = args['test_data_csv']
    DATA_PATH = args['data_path']
    print('data path is {}'.format(DATA_PATH))
       
    device_type = 'gpu'
    #gpu_id = 0
    gpu_id = args['gpuid']
    threshold = 0.5
    max_value = 65536
    # max_value = 255

    num_chan = args['num_chan']

    # datasets = [Train_data, Val_data, Test_data]
    os.makedirs(output_path, exist_ok=True)

    parameters = {
        "device_type": device_type,
        "gpu_number": gpu_id,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        # model related hyper-parameters
        "percent_t": percent_t,
        "cam_size": (46, 30),
        # "cam_size": (92, 60),
        "K": 6,
        "crop_shape": (256, 256),
        # "use_v1_global":False
        "use_v1_global":use_v1_global,
        "pretrain_local":pretrain_local,
        "pretrain_global":pretrain_local,
    }

    if use_v1_global:
        parameters["cam_size"] = (92, 60)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # import pdb;pdb.set_trace()

    model = gmic.GMIC(parameters)
    # model_path = pretrained_state_dict
    if device_type == 'gpu' and torch.has_cudnn:
    #     model.load_state_dict(torch.load(model_path), strict=False)
        device = torch.device("cuda:{}".format(gpu_id))
    else:
    #     model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        device = torch.device("cpu")
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)


    with open(os.path.join(output_path, 'training_info-{}_bs{}_ep{}.csv'.format(output_path.split(os.sep)[1].split('.')[0], batch_size, epochs)), mode='w') as csv_file:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'train_auc', 'train_sen', 'train_spe', 'val_loss', 'val_acc', 'val_auc', 'val_sen', 'val_spe']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        train_loss, train_acc, valid_loss, valid_acc = train_net(model, output_path, epochs, img_size, batch_size, optimizer, lr_scheduler, \
            DATA_PATH, device, threshold, writer, beta, lr_rate, max_value,aug,num_chan)
        # plot training curve
        N = [n+1 for n in range(epochs)]

        # summarize history for loss
        fig = plt.figure()
        plt.plot(N, train_loss, label='train_loss')
        plt.plot(N, valid_loss, label='valid_loss')
        plt.title('Training loss curve of mammo classifier')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plotname = 'Training loss curve-{}_bs{}_epoch{}.jpg'.format(output_path.split(os.sep)[1].split('.')[0], batch_size, epochs)
        plt.savefig(os.path.join(output_path, plotname))
        plt.close(fig)

        fig = plt.figure()
        plt.plot(N, train_acc, label='train_accuracy')
        plt.plot(N, valid_acc, label='valid_accuracy')
        plt.title("Training accuracy curve of mammo classifier")
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plotname = 'Training accuracy curve-{}_bs{}_epoch{}.jpg'.format(output_path.split(os.sep)[1].split('.')[0], batch_size, epochs)
        plt.savefig(os.path.join(output_path, plotname))
        plt.close(fig)
        
        fig = plt.figure()
        plt.plot(N, train_acc, label='train_auc')
        plt.plot(N, valid_acc, label='valid_auc')
        plt.title("Training auc curve of mammo classifier")
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plotname = 'Training auc curve-{}_bs{}_epoch{}.jpg'.format(output_path.split(os.sep)[1].split('.')[0], batch_size, epochs)
        plt.savefig(os.path.join(output_path, plotname))
        plt.close(fig)