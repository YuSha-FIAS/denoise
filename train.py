from denoise_net import Net
import torch
import argparse
from torch.utils.data import DataLoader
from data_loader import MyDataset
import matplotlib as plt
from torch.nn import L1Loss

parser = argparse.ArgumentParser(description='denoise ')
parser.add_argument('--cuda', action='store_true', help='Choose device to use cpu cuda:0')
parser.add_argument('--batch_size', action='store', type=int,
                        default=1, help='number of data in a batch')
parser.add_argument('--lr', action='store', type=float,
                        default=0.0001, help='initial learning rate')
parser.add_argument('--epochs', action='store', type=int,
                        default=200, help='train rounds over training set')

def train(opts):
    # device = torch.device('cpu') if not torch.cuda.is_available or opts.cpu else torch.device('cuda')
    device = torch.device("cuda")
    print(device)
    # load dataset
    dataset_train = MyDataset('train.txt')
    dataset_test = MyDataset('test.txt')
    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=opts.batch_size, shuffle=True, num_workers=0)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=opts.batch_size, shuffle=False, num_workers=0)
    model = Net()
    # model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=opts.lr,betas=(0.9,0.99))
    loss_fct = L1Loss()   #
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    train_loss_list = []
    test_loss_list = []
    for epoch in range(opts.epochs):
        train_batch_num = 0
        train_loss = 0.0
        model.train()
        for img, label in data_loader_train:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = loss_fct(pred, label)
            loss.backward()
            optimizer.step()
            train_batch_num += 1
            train_loss += loss.item()

        train_loss_list.append(train_loss / len(data_loader_train.dataset))
        model.eval()
        test_loss = 0
        test_batch_num = 0

        with torch.no_grad():
            for test_img, test_label in data_loader_test:
                test_img = test_img.to(device)
                test_label = test_label.to(device)
                t_pred = model(test_img)
                # accuracy
                loss = loss_fct(t_pred, test_label)
                test_loss += loss.item()
                test_batch_num += 1

        test_loss_list.append(test_loss / len(data_loader_test.dataset))
        print('epoch: %d, train loss: %.4f, test loss: %.4f' %
              (epoch, train_loss / train_batch_num, test_loss/ test_batch_num))

if __name__ == "__main__":
    opts = parser.parse_args() # Namespace object
    train(opts)

