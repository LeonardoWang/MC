from __future__ import print_function
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import sys
sys.path.append('../')
from config import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        """module, pruner_ratio, pruner_dimension = None, pruner_name = None"""
        """@nni.compression.weight_pruning()"""
        Compression().param_mask(self.conv1.weight , 0.5, None, 'PytorchLevelParameterPruner')
        """@nni.compression.quantize(self.conv1)"""
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        

        #for module_full_name, module in self.named_parameters():
        #    print('self parameters',module_full_name,type(module))

        

    def forward(self, x):
        #print(x.size())
        #use_mask(self.conv1.weight,0.5,0)
        '''
        self.conv1.weight = func(self.conv1.weight, 0.5, epoch_num, minibatch_num):
            if epoch_num > 5:
                indice = min(self.conv1.weight)
                mask = gen_mask(indice)
                return self.conv1.weight * mask
        '''
        x = F.relu(self.conv1(x))
        #print('type',type(x),x.dtype,x.to(dtype=torch.float16).dtype)
        #print(x.size(),type(self.conv1.name),self.conv1.named_parameters())
        
        #print(torch.norm(x,p=0)/torch.numel(x))
        x = F.max_pool2d(x, 2, 2)
        #print(x.size())
        #x = use_mask(x,0.5,1)
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = F.max_pool2d(x, 2, 2)
        #print(x.size())
        #x = use_mask(x,0.5,2)
        x = x.view(-1, 4*4*50)
        #print(x.size())
        x = F.relu(self.fc1(x))
        #print(x.size())
        x = self.fc2(x)
        #print(x.size())
        #print(torch.norm(x,p=0)/torch.numel(x))
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        Compression().apply_mask(model,epoch)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    
    #model.load_state_dict(torch.load('mnist_cnn.pt',map_location='cpu'))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    level = {}
    count = 0
    for name, param in model.named_parameters():
        print('test_mnist_torch', name)
        level[name] = 0.5
        count += 1
        if count == 2:
            break
    '''
    scheduler = CompressionScheduler(model)
    pruner = PytorchLevelParameterPruner(level)
    policy = PytorchPruningPolicy(pruner,level)
    scheduler.addPolicy(policy,None,1,10)
    '''
    #scheduler = MC.config.create_scheduler(model, 'pytorch', 'PytorchLevelParameterPruner',level)
    
    for epoch in range(1, args.epochs + 1):
        for name,param in model.named_parameters():
            if name in level:
                pass
                #print(param.data)
        
        """@nni.compresson.on_epoch_begin()"""
        for name,param in model.named_parameters():
            if name in level:
                pass
                #print(param.data)
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    test(args, model, device, test_loader)
    
   


        
if __name__ == '__main__':
    main()
