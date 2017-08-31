import torch
from torchvision import datasets, transforms

def load_data(train_path, val_path=None):
	if val_path is None:
		val_path = train_path

	train_transfrom = transforms.Compose([
						transforms.Scale(240),
						transforms.RandomSizedCrop(224),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
					])

	val_transfrom = transforms.Compose([
						transforms.Scale(240),
						transforms.CenterCrop(224),
						transforms.ToTensor(),
						transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
					])

	dsets = {
		'train': datasets.ImageFolder(train_path, train_transfrom),
		'val': datasets.ImageFolder(val_path, val_transfrom)
	}

	dset_loaders = {
		x: torch.utils.data.DataLoader(dsets[x], batch_size=7, shuffle=True, num_workers=4)
		for x in ['train', 'val']
	}

	dset_sizes = {
		x: len(dsets[x])
		for x in ['train', 'val']
	}

	dset_classes = dsets['train'].classes

	return dset_loaders, dset_sizes, dset_classes
