import torch
from torch.autograd import Variable
import time
import copy
import pickle

def train(model, criterion, optimizer, lr_scheduler, 
            dsets_loader, dsets_sizes, lr, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0

    history = {
        x: []
        for x in ['train', 'val']
    }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print ('-' * 10)

        for phase in ['train', 'val']:
            start = time.time()

            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch, init_lr=lr)
                model.train(True)
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for data in dsets_loader[phase]:
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                optimizer.zero_grad()

                if phase == 'train':
                    o1, o2, o3 = model(inputs)
                    _, preds = torch.max(o1.data, 1)

                    loss1 = criterion(o1, labels)
                    loss2 = criterion(o2, labels)
                    loss3 = criterion(o3, labels)

                    loss = loss1 + 0.3 * (loss2 + loss3)

                else:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dsets_sizes[phase]
            epoch_acc = running_corrects / dsets_sizes[phase]
            epoch_time = time.time() - start

            state = {
                'phase': phase,
                'epoch': epoch,
                'epoch_loss': epoch_loss,
                'epoch_acc': epoch_acc,
                'lr': optimizer.state_dict()['param_groups'][0]['lr']
            }

            history[phase].append(state)

            print ('{} Loss: {:.4f} Acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(
                    phase, epoch_loss, epoch_acc,
                    epoch_time // 60, epoch_time % 60))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
        print ()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    print('Saving history')
    with open('history.pickle', 'wb') as f:
        pickle.dump(history, f)

    return best_model, best_acc

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=8):
    lr = init_lr * (0.96**(epoch // lr_decay_epoch))
    
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer