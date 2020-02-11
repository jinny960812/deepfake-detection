Learn more or give us feedback
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from dataset.mydataset import Dataset_CRNN
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from network.ResNetCRNN import ResCNNEncoder, DecoderRNN
from tqdm import tqdm




def main():
    global cnn_encoder
    args = parse.parse_args()
    data_path = args.data_path
    continue_train = args.continue_train
    saved_model_path = args.saved_model_path
    batch_size = args.batch_size
    dropout_p = args.dropout_p
    num_frames = args.num_frames
    name = args.name
    output_path = os.path.join('./output', name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    CNN_embed_dim = 512
    RNN_hidden_layers = 3
    RNN_hidden_nodes = 256
    RNN_FC_dim = 128
    frame_list = np.arrange(0, num_frames, 10).tolist()

    #data loading paramters
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    #set data loader
    train_dataset = Dataset_CRNN(data_path=data_path, frame_list=frame_list, transform=xception_default_data_transforms['test'] )
    train_loader = data.DataLoader(train_dataset, **params)

    #load CRNN model
    cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim ).to(device)
    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=2).to(device)

    #load state dictionary
    if continue_train:
        cnn_encoder.load_state_dict(torch.load(os.path.join(saved_model_path, 'cnn_encoder.pth')))
        rnn_decoder.load_state_dict(torch.load(os.path.join(saved_model_path, 'cnn_decoder.pth')))
        print('CRNN model loaded!')

    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

    cnn_encoder.train()
    rnn_decoder.train()

    model_parameters = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    corrects = 0
    acc = 0
    best_model_cnn = cnn_encoder.state_dict()

    print('Training all {} videos:'.format(test_dataset_size))

    with torch.no_grad():
        iter_corrects=0
        train_loss = 0.0
        train_corrects = 0.0
        for i, (X, label) in enumerate (tqdm(train_loader)):
            X = X.to(device)
            labels = labels.to(device)

            output = rnn_decoder(cnn_encoder(X))
            _, preds = torch.max(output.data,1)


            corrects += torch.sum(preds == labels.data).to(torch.float32)
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)

            if not (i % 100):
                print('iteration {} Acc: {:.4f}'.format(i, iter_corrects / batch_size))
                print('cumulative corrects {}'.format(corrects))

        print("...............................")
        acc = corrects/test_dataset_size
        print('Test Acc: {:.4f}'.format(acc))




if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--continue_train', type=bool, default=False)
    parse.add_argument('--name', '-n', type=str, default='rescrnn')
    parse.add_argument('--num_frames', '-nf', type=int, default=300)
    parse.add_argument('--data_path', '-bp', type=str, default='./dataset')
    parse.add_argument('--saved_model_path', '-sm', type=str, default='./ResNetCRNN_ckpt')
    parse.add_argument('--batch_size', '-bs', type=int, default=16)
    parse.add_argument('--dropout_p', 'd', type=int, default=0.3)
    main()
