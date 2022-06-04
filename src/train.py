import torch
import torch.optim as optim
import sys
from model import RregressionModel
import torchvision
import argparse
from tqdm import tqdm
import Dataset

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--root', default='..\\dataset')  # path to dataset
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--N', default=1)  # Threshold used to judge prediction results

parser.add_argument('--data_arg',
                    default=True)  # data augmentation, set to True when training, set to False when testing
parser.add_argument('--ASM', default=True)  # Whether to use ASM  in the model, True represents yes, False represents no
parser.add_argument('--ev', default=2.2)  # expected variance in ASM, default=2.2

parser.add_argument('--cs', default=False)  # Calculate the scaling factor, set to True when using (ASM must also be set to True)
parser.add_argument('--sf', default=0)  # Scaling factor, which takes effect when ASM=False


args = parser.parse_args()


def load_model(name='new'):  # Load dataset:Change name='old' to load a saved model
    if type(name) != str:
        sys.exit('Command must be a string!')
    if name == 'new':
        model = RregressionModel()
    elif name == 'old':
        model = torch.load(
            '..\\model_save\\xxx.pkl')  # old model path
    else:
        sys.exit('Command must be new or old!')
    return model


def compute_scalingfactor(mode, dataloader, threshold=1):
    mode.eval()
    correct = 0
    total = 0
    batch_size = 0
    total_fact = 0.0  # sum of n batch Scaling factor
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, labels, typ = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)[0]
            # Calculate the sum of n batch Scaling factor
            factors = model(images)[1]
            total_fact = total_fact + factors

            batch_size += 1
            total += labels.size(0)
            # Set the threshold N to determine the prediction result
            a = outputs.cpu().numpy()  # predicted value
            b = labels.cpu().numpy()  # label
            correctlist = [abs(a[i] - b[i]) for i in range(0, len(a))]
            for i in range(len(correctlist)):
                if correctlist[i] <= threshold:
                    correct += 1
    print('Accuracy : %.2f %%' % (100 * correct / total))
    print('Scaling factor: %.10f' % (round(total_fact / batch_size, 10)))


def train(epoch, train_load, mode):
    mode.train()
    running_loss = 0.0
    total_loss = 0.0
    batch_size = 0

    for batch_idx, data in tqdm(enumerate(train_load, 0)):
        inputs, target, _ = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).to(torch.float32)
        # type conversion
        target = target.to(torch.float32)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()
        batch_size += 1
        if batch_idx % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0
    return total_loss / batch_size


def test(mode, test_load, threshold=1):
    mode.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_load):
            images, labels, typ = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total += labels.size(0)
            # Set the threshold N to determine the prediction result
            a = outputs.cpu().numpy()  # predicted value
            b = labels.cpu().numpy()  # label
            correctlist = [abs(a[i] - b[i]) for i in range(0, len(a))]
            for i in range(len(correctlist)):
                if correctlist[i] <= threshold:
                    correct += 1
    print('Accuracy on val set: %.2f %%' % (100 * correct / total))


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    model = load_model()
    model.asm, model.scalingfactor, model.cs, model.expertvar = args.ASM, args.sf, args.cs, args.ev
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader, test_loader, val_loader = Dataset.train_test_dataloader(args.root, args.batch_size, args.num_workers,
                                                                          args.data_arg)

    # Define the optimizer and loss function
    criterion = torch.nn.HuberLoss(delta=1)
    optimizer = optim.AdamW(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=6, verbose=True,
                                                           min_lr=5.0e-6)
    # Train-----------------------------
    for epoch in range(args.epoch):
        if model.cs:
            compute_scalingfactor(model, train_loader, args.N)
        else:
            train_loss = train(epoch, train_loader, model)
            print('[%d] loss: %.3f' % (epoch + 1, train_loss))
            # Update the learning rate according to lr and print the current learning rate
            scheduler.step(train_loss)
            print('lr of next epoch:' + str(optimizer.state_dict()['param_groups'][0]['lr']))
            # save model
            torch.save(model,
                       '../model_save/model_epoch' + str(
                           epoch) + '.pkl')
    # test(model, test_loader, args.N) # test

