import torch, torchvision, torch.nn as nn
from torch import Tensor
import numpy as np
import time
from torchvision import datasets, transforms
import sys
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from . import MixedNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'



## Main training functions

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''



def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

term_width = 32

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def load_data(data_dir='/content/image', seed=42, train_rate=0.5):
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset_train = datasets.ImageFolder(data_dir, transform=transform_train)
    dataset_test  = datasets.ImageFolder(data_dir, transform=transform_test)

    num_train = int(len(dataset_train) * train_rate)
    num_test = len(dataset_train) - num_train

    trainset = torch.utils.data.random_split(dataset_train, [num_train,num_test], generator=torch.Generator().manual_seed(seed))[0]
    testset  = torch.utils.data.random_split(dataset_test,  [num_train,num_test], generator=torch.Generator().manual_seed(seed))[1]

    return trainset, testset


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    return correct/total


def test(epoch):
    global best_acc, state
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc >= best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    
    return correct/total
  

## Training Loop

def plot_results(num_epochs):
    x = [i for i in range(num_epochs)]
    plt.plot(range(num_epochs), train_acc, 'b', label='Training')
    plt.plot(range(num_epochs), test_acc, 'g', label='Test')
    plt.grid()
    plt.legend()
    plt.show()

def train_loop(model, num_epochs, k_fold_number, learning_rate, batch_size, weight_decay):
    global best_acc, optimizer, criterion, scheduler, net, trainloader, testloader, train_acc, test_acc
    results = []
    for k in range(1,k_fold_number+1):
        print(f"STARTING VALIDATION {k}/{k_fold_number}")
        best_acc = 0  # best test accuracy
        start_epoch = 0 
        
        trainset, testset = load_data(seed=k, train_rate=0.5)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Model
        print('==> Building model..')
        net = model()
        net = net.to(device)
        # if device == 'cuda':
        #     net = torch.nn.DataParallel(net)
        #     cudnn.benchmark = True
        if k==1:
            torch.save(net.state_dict(), '/content/drive/MyDrive/IC/params.pt')
        else:
            net.load_state_dict(torch.load('/content/drive/MyDrive/IC/params.pt')) # garantindo que a rede vai iniciar com mesmos parametros em cada uma das 10 iteracoes
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=True)

        # Treina o modelo
        train_acc = np.zeros(num_epochs)
        test_acc = np.zeros(num_epochs)
        for epoch in range(start_epoch, start_epoch+num_epochs):
            train_acc[epoch] = train(epoch)*100.
            test_acc[epoch] = test(epoch)*100.
            scheduler.step()

        results.append(state)
        plot_results(num_epochs)
    
    return results # lista com k_fold_number states, cada um contendo state_dict, accuracy e epoch do melhor modelo daquela validacao
  
  
  
res = train_loop(model=MixedNet, num_epochs=15, k_fold_number=10, learning_rate=1e-3, batch_size=16, weight_decay=5e-4)
  
results = []
epochs = []
for state in res:
    results.append(state['acc'])
    epochs.append(state['epoch'])
results, epochs

media, std = np.mean(results), np.std(results)
print('Test Accuracy (%):')
print('Mean: %.2f' %media)
print('STD: %.2f' %std)
