import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


cuda = torch.device("cuda:0")

class Hsigmoid(nn.Module):                                                                                                                                                                  
     def __init__(self, inplace=True):                                                                                                                                                       
         super(Hsigmoid, self).__init__()                                                                                                                                                    
         self.inplace = inplace                                                                                                                                                              
                                                                                                                                                                                             
     def forward(self, x):                                                                                                                                                                   
         return F.relu6(x + 3., inplace=self.inplace) / 6.
     
class SEModule(nn.Module):                                                                                                                                                                  
    def __init__(self, channel, reduction=4):                                                                                                                                               
        super(SEModule, self).__init__()                                                                                                                                                    
        self.avg_pool = nn.AdaptiveAvgPool2d(1)                                                                                                                                             
        self.fc = nn.Sequential(                                                                                                                                                            
            nn.Linear(channel, channel // reduction, bias=False),                                                                                                                           
            nn.ReLU(inplace=True),                                                                                                                                                          
            nn.Linear(channel // reduction, channel, bias=False),                                                                                                                           
            Hsigmoid()                                                                                                                                                                      
        )                                                                                                                                                                                   
                                                                                                                                                                                            
    def forward(self, x):                                                                                                                                                                   
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  
        y = y.cuda()    
        self.fc.cuda()                                                                                                                                             
        y = self.fc(y).view(b, c, 1, 1)                                                                                                                                                  
        return x * y.expand_as(x)

class Hswish(nn.Module):                                                                                                                                                                    
    def __init__(self, inplace=True):                                                                                                                                                       
        super(Hswish, self).__init__()                                                                                                                                                      
        self.inplace = inplace                                                                                                                                                              
                                                                                                                                                                                  
    def forward(self, x):                                                                                                                                                                   
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


'Defining Fuse network'

class Fuse_Net(nn.Module):
    def __init__(self, channels, kernel_size, exp_size, out_size, SE, NL, stride):
        
        self.c = channels
        self.k = kernel_size
        self.e = exp_size
        self.o = out_size
        self.SE = SE
        self.NL = NL
        self.stride = stride
        self.pr = int((self.k[0]-1)/2)
        self.ps = int((self.k[1]-1)/2)
        
        super(Fuse_Net, self).__init__()
        
        self.conv1 = nn.Conv2d(self.c, self.e, 1,bias=False)
        self.conv1_bn = nn.BatchNorm2d(self.e)
        
        self.conv_f1 = nn.Conv2d(self.e, self.e, self.k, stride = self.stride, padding = (self.pr, self.ps ),bias=False)
        self.conv_f1_bn = nn.BatchNorm2d(self.e)
        
        self.k.reverse()
        self.conv_f2 = nn.Conv2d(self.e, self.e, self.k, stride = self.stride, padding = (self.ps, self.pr ),bias=False)
        self.conv_f2_bn = nn.BatchNorm2d(self.e)
        
        self.conv2 = nn.Conv2d(2*(self.e), self.o, 1, bias=False)
        
             
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv1_bn(x)
        if self.NL == 0:
            x = F.relu(x)
        else:
            hswish = Hswish()
            x = hswish.forward(x)
        
        x1 = self.conv_f1(x)
        x1 = self.conv_f1_bn(x)
        
        x2 = self.conv_f2(x)
        x2 = self.conv_f2_bn(x)
        
        x = torch.cat([x1,x2],1)
        
        if self.SE == 1:
            sem = SEModule(2*self.e)
            x = sem.forward(x)
        hsig = Hsigmoid()
        x  = hsig.forward(x)
        
        if self.NL == 0:
            x = F.relu(x)
        else:
            hswish = Hswish()
            x = hswish.forward(x)
            
        x = self.conv2(x)
        
        
        
        return x


