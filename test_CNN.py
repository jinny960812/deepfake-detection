import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
def main():
	#initialize arguments
	args = parse.parse_args()
	test_list = args.test_list
	batch_size = args.batch_size
	model_path = args.model_path
	torch.backends.cudnn.benchmark=True

	#set data loader
	test_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['test'])
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

	#model = torchvision.models.densenet121(num_classes=2)
	#model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)

	#load model and state dictionary
	model = model_selection(modelname='resnet18', num_out_classes=2, dropout=0.5)
	model.load_state_dict(torch.load(model_path))
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	model = model.cuda()
	model.eval()

	test_dataset_size = len(test_dataset)
	corrects = 0
	acc = 0
	iteration = 0
	with torch.no_grad():
		for (image, labels) in test_loader:
			iter_corrects=0
			image = image.cuda()
			labels = labels.cuda()

			outputs = model(image)
			_, preds = torch.max(outputs.data, 1)

			corrects += torch.sum(preds == labels.data).to(torch.float32)
			iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
			iteration+=1
			if not (iteration % 100):
				print('iteration {} Acc: {:.4f}'.format(iteration, iter_corrects / batch_size))
				print(iter_corrects)
				print(corrects)

			#print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/batch_size))
		acc = corrects / test_dataset_size
		print('Test Acc: {:.4f}'.format(acc))
		print(test_dataset_size)



if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--batch_size', '-bz', type=int, default=16)
	parse.add_argument('--test_list', '-tl', type=str, default='./List_testing_1.txt')
	parse.add_argument('--model_path', '-mp', type=str, default='./pretrained_model/df_c0_best.pkl')
	main()
	print('Hello world!!!')