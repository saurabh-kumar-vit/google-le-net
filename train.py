import data_util
import model
import train_util

import torch
import torch.nn as nn
import torch.optim as optim

train_path = ''
val_path = ''

dset_loaders, dset_sizes, dset_classes = data_util.load_data(train_path=train_path, 
                                                            val_path=val_path)

print(dset_sizes)
print(dset_classes)


net = model.GoogleLeNet().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), weight_decay=0.0005)
lr_scheduler = train_util.exp_lr_scheduler
lr = 0.001

best_model, best_acc = train_util.train(net, criterion, optimizer, lr_scheduler,
                                dset_loaders, dset_sizes, lr, 40)

print('Saving the best model')
filename = 'trained_model_val_{:.2f}.pt'.format(best_acc)
torch.save(best_model.state_dict(), filename)
