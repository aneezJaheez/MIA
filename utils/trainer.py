import torch
import torch.nn as nn
import os
import pickle

from datetime import datetime
from torch.utils.data import DataLoader

def store_attack_sets(model, trainloader, testloader, out_dir, device, num_samples=None):
    
    #Attack data will be stored as a dictionary, with the key being the class, and the value being an inner dictionary
    #The inner dictionary for each class will contain the data itself, already transformed, the output probabilities
    
    out_file = "attack_data_train"
    out_path = os.path.join(out_dir, out_file)
    
    if os.path.exists(out_path):
        with open(out_path, "rb") as infile:
            attack_data = pickle.load(infile, encoding="latin1")
    else:
        attack_data = {}
        for i in range(10):
            #label = 1 for in, 0 for out
            attack_data[i] = {
                "data" : [],
                "labels" : [],
            }

    model.eval()

    with torch.no_grad():
        total = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            posteriors = model(inputs)

            #Input is of shape (batch_size x 3 x 32 x 32)
            #Posterior is of shape (batch_size x 10)
            for i, probs in enumerate(posteriors):
                attack_data[targets[i].item()]["data"].append(posteriors[i].cpu())
                attack_data[targets[i].item()]["labels"].append(1)

            total += inputs.size(0)

            if num_samples is None:
                continue
            elif total >= num_samples:
                break

        total = 0
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            posteriors = model(inputs)

            #Input is of shape (batch_size x 3 x 32 x 32)
            #Posterior is of shape (batch_size x 10)
            for i, probs in enumerate(posteriors):
                attack_data[targets[i].item()]["data"].append(posteriors[i].cpu())
                attack_data[targets[i].item()]["labels"].append(0)

            total += inputs.size(0)

            if num_samples is None:
                continue
            elif total >= num_samples:
                break

    with open(out_path, "wb") as outfile:
        pickle.dump(attack_data, outfile)

    print("Saved attack data")
    return
        

def test_step(model, testloader, criterion, epoch, device, pred_threshold=0.5):
    model.eval()
    test_loss, test_acc = 0., 0.
    correct, total = 0, 0

    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            if outputs.size(1) == 1:
                #BCE Loss
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, targets.float())
                preds = (outputs >= pred_threshold).float()
            else:
                #CE Loss
                loss = criterion(outputs, targets)
                _, preds = outputs.max(dim=1)

            test_loss += loss.item()
            total += data.size(0)
            correct += preds.eq(targets).sum().item()

    test_acc = 100. * correct / total
    test_loss /= len(testloader)

    print('[Test]  Epoch: {}\tTest CE Loss: {:.6f}\tTest Acc.: {:.1f}%'.format(epoch, test_loss, test_acc))
    return test_loss, test_acc


def train_step(model, trainloader, optimizer, criterion, epoch, device, pred_threshold=0.5):
    model.train()
    train_loss, train_acc = 0., 0.
    correct, total = 0, 0

    for data, targets in trainloader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        if outputs.size(1) == 1:
            #BCE Loss
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, targets.float())
            preds = (outputs > 0.5).float()
        else:
            #CE Loss
            loss = criterion(outputs, targets)
            _, preds = outputs.max(dim=1)

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        total += data.size(0)
        correct += preds.eq(targets).sum().item()

    train_acc = 100. * correct / total
    train_loss /= len(trainloader)

    print('[Train]  Epoch: {}\tTrain CE Loss: {:.6f}\tTrain Acc.: {:.1f}%'.format(epoch, train_loss, train_acc))
    return train_loss, train_acc


def train_model(model, trainset, testset, optimizer, criterion, num_epochs, 
                batch_size=64, device=torch.device("cpu"), checkpoint_dir=None, 
                log_file=None, data_out_dir=None, num_workers=0, bce_pred_threshold=0.5):
    
    run_id = str(datetime.now())

    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )

    testloader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    with open(log_file, "a") as af:
        columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
        af.write('\t'.join(columns) + '\n')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

    best_train_acc, train_acc = -1, -1
    best_test_acc, test_acc = -1, -1

    for epoch in range(num_epochs):
        train_loss, train_acc = train_step(model, trainloader, optimizer, criterion, epoch+1, device, bce_pred_threshold)
        best_train_acc = max(best_train_acc, train_acc)

        test_loss, test_acc = test_step(model, testloader, criterion, epoch+1, device, bce_pred_threshold)
        best_test_acc = max(best_test_acc, test_acc)

        if test_acc >= best_test_acc:
            state = {
                "epoch": epoch,
                "arch": model.__class__,
                "state_dict": model.state_dict(),
                "best_acc": test_acc,
                "optimizer": optimizer.state_dict(),
                "created_on":  run_id,
            }
            torch.save(state, checkpoint_path)

        with open(log_file, 'a') as af:
            train_cols = [run_id, epoch + 1, 'train', train_loss, train_acc, best_train_acc]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            test_cols = [run_id, epoch + 1, 'test', test_loss, test_acc, best_test_acc]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    print("Training Completed")

    if data_out_dir is not None:
        store_attack_sets(model, trainloader, testloader, out_dir = data_out_dir, device=device)

    return model