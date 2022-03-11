from torchvision import datasets
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import argparse
import pickle
import torch


from model import *


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def test(model_path : nn.Module,
         test_dataset_path : str, 
         batch_size : int = 3, 
         use_cuda : bool = False):
	model = torch.load(model_path, map_location=torch.device('cpu'))
	classes = ('plane', 
		'car', 
		'bird', 
		'cat', 
		'deer', 
		'dog', 
		'frog', 
		'horse', 
		'ship', 
		'truck'
		)
	means = [0.4914, 0.4822, 0.4465]
	stds = [0.247, 0.243, 0.261]

	test_dataset = ImageFolderWithPaths(test_dataset_path, 
                                        transform=transforms.Compose([
                                            transforms.ToTensor(), 
                                            transforms.Normalize((means[0], means[1], means[2]), 
                                                                 (stds[0], stds[1], stds[2]))]))
	dataloader = torch.utils.data.DataLoader(test_dataset, 
                                             batch_size = batch_size, 
                                             shuffle = False, 
                                             num_workers = 100)
	path_to_class = {}
	model.train = False
	with torch.no_grad():
		for X_batch, _, _paths in dataloader:
			torch.cuda.empty_cache()
			var = Variable(torch.FloatTensor(X_batch))
			if use_cuda:
				logits = model.cuda()(var.cuda())
			else:
				logits = model(var)
			y_pred = logits.max(1)[1].data
			for i in range(len(_paths)):
				path_to_class[_paths[i]] = classes[y_pred[i]]
	return path_to_class


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", help="path to the model")
	parser.add_argument("--test_dataset_path", help="path to the folder with images to predict classes")
	parser.add_argument("--batch_size", help="predicting dataloader batch_size, default=3")
	parser.add_argument("--use_cuda", help="1 if cuda is used, 0 otherwise, default=0")
	args = parser.parse_args()
	if args.model_path and args.test_dataset_path:
	    batch_size = 3
	    use_cuda = 0
	    if args.batch_size:
	    	batch_size = int(args.batch_size)
	    if args.use_cuda:
	    	use_cuda = int(args.use_cuda)
	    result = test(args.model_path, args.test_dataset_path, batch_size, use_cuda)
	    with open("result.txt", 'w') as f:
	    	for key in result.keys():
	    		f.write(key + " : " + str(result[key] + "\n"))
	    print("Result is written into result.txt")
	else:
		print("ERROR! model_path and test_dataset_path must be specified!")
