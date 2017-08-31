import torch.nn as nn
import torch
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class _Stem(nn.Module): 
    def __init__(self):
        super(_Stem, self).__init__()
        self.stem = nn.Sequential(
                        BasicConv2d(3, 64, kernel_size=7, padding=2, stride=2),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        BasicConv2d(64, 64, kernel_size=1),
                        BasicConv2d(64, 192, kernel_size=3, padding=1),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    )

    def forward(self, x):
        x = self.stem(x)

        return x

class _Auxiliary_Classifiers(nn.Module):
    def __init__(self, num_fts):
        super(_Auxiliary_Classifiers, self).__init__()
        self.aux_classifier = nn.Sequential(
                                    nn.AvgPool2d(kernel_size=5, stride=3, padding=2),
                                    BasicConv2d(num_fts, 128, kernel_size=1),
                                )

        self.classifier = nn.Sequential(
                                nn.Linear(5 * 5 * 128, 1024),
                                nn.Dropout(p=0.7),
                                nn.Linear(1024, 2)
                            )

    def forward(self, x):
        x = self.aux_classifier(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

class _Inception_Module(nn.Module):
    def __init__(self, conv1x1, conv1x1_reduce_3, conv1x1_reduce_5,
                        conv3x3, conv5x5, conv1x1_reduce_pool,
                        num_fts):
        super(_Inception_Module, self).__init__()

        self.conv1x1_branch = nn.Sequential(
                                    BasicConv2d(num_fts, conv1x1, kernel_size=1),
                                )

        self.conv3x3_branch = nn.Sequential(
                                    BasicConv2d(num_fts, conv1x1_reduce_3, kernel_size=1),
                                    BasicConv2d(conv1x1_reduce_3, conv3x3, kernel_size=3, padding=1),
                                )

        self.conv5x5_branch = nn.Sequential(
                                    BasicConv2d(num_fts, conv1x1_reduce_5, kernel_size=1),
                                    BasicConv2d(conv1x1_reduce_5, conv5x5, kernel_size=5, padding=2),
                                )

        self.pool_branch = nn.Sequential(
                                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                    BasicConv2d(num_fts, conv1x1_reduce_pool, kernel_size=1),
                                )

    def forward(self, x):
        x1x1 = self.conv1x1_branch(x)
        x3x3 = self.conv3x3_branch(x)
        x5x5 = self.conv5x5_branch(x)
        xpool = self.pool_branch(x)

        x = torch.cat([x1x1, x3x3, x5x5, xpool], 1)

        return x 


class GoogleLeNet(nn.Module):
    def __init__(self):
        super(GoogleLeNet, self).__init__()
        self.features1 = nn.Sequential(
                            _Stem(),
                            _Inception_Module(conv1x1=64, conv1x1_reduce_3=96, conv3x3=128, 
                                conv1x1_reduce_5=16, conv5x5=32, conv1x1_reduce_pool=32, 
                                num_fts=192),
                            _Inception_Module(conv1x1=128, conv1x1_reduce_3=128, conv3x3=192, 
                                conv1x1_reduce_5=32, conv5x5=96, conv1x1_reduce_pool=64, 
                                num_fts=256),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                            _Inception_Module(conv1x1=192, conv1x1_reduce_3=96, conv3x3=208, 
                                conv1x1_reduce_5=16, conv5x5=48, conv1x1_reduce_pool=64, 
                                num_fts=480)
                        )

        self.features2 = nn.Sequential(
                            _Inception_Module(conv1x1=160, conv1x1_reduce_3=112, conv3x3=224, 
                                conv1x1_reduce_5=24, conv5x5=64, conv1x1_reduce_pool=64, 
                                num_fts=512),
                            _Inception_Module(conv1x1=128, conv1x1_reduce_3=128, conv3x3=256, 
                                conv1x1_reduce_5=24, conv5x5=64, conv1x1_reduce_pool=64, 
                                num_fts=512),
                            _Inception_Module(conv1x1=112, conv1x1_reduce_3=144, conv3x3=288, 
                                conv1x1_reduce_5=32, conv5x5=64, conv1x1_reduce_pool=64, 
                                num_fts=512)
                        )

        self.features3 = nn.Sequential(
                            _Inception_Module(conv1x1=256, conv1x1_reduce_3=160, conv3x3=320, 
                                conv1x1_reduce_5=32, conv5x5=128, conv1x1_reduce_pool=128, 
                                num_fts=528),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                            _Inception_Module(conv1x1=256, conv1x1_reduce_3=160, conv3x3=320, 
                                conv1x1_reduce_5=32, conv5x5=128, conv1x1_reduce_pool=128, 
                                num_fts=832),
                            _Inception_Module(conv1x1=384, conv1x1_reduce_3=192, conv3x3=384, 
                                conv1x1_reduce_5=48, conv5x5=128, conv1x1_reduce_pool=128, 
                                num_fts=832)
                        )
        
        self.Aux1 = _Auxiliary_Classifiers(num_fts=512)
        self.Aux2 = _Auxiliary_Classifiers(num_fts=528)

        self.fc = nn.Linear(1024, 2)
    def forward(self, x):
        x = self.features1(x)

        if self.training:
            aux1 = self.Aux1(x)

        x = self.features2(x)

        if self.training:
            aux2 = self.Aux2(x)

        x = self.features3(x)
        x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(-1 ,1024)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.fc(x)
        
        if self.training:
            return x, aux1, aux2
        else:
            return x
