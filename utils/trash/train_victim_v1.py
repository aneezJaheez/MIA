import torch
import os

import os.path as osp
import torch.optim as optim


from torch.utils.data import DataLoader
from datetime import datetime
import torch.nn.functional as F

def test_step(model, test_loader, criterion, device, epoch):
    model.eval()

    test_loss, correct, total = 0., 0, 0
    ground_truth_history = dict()
    correct_history = dict()
    predict_history = dict()

    batch_idx = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

            for log_set_index, probs in enumerate([targets, predicted, predicted[predicted == targets]]):
                if log_set_index == 0:
                    log_set = ground_truth_history
                elif log_set_index == 1:
                    log_set = predict_history
                else:
                    log_set = correct_history

                labels, indices = torch.unique(probs, sorted=True, return_counts = True)
                for i, v in enumerate(labels):
                    v = v.item()
                    if v not in log_set:
                        log_set[v] = indices[i].item()
                    else:
                        log_set[v] += indices[i].item()

    acc = 100. * correct / total
    test_loss /= (batch_idx + 1)

    print(f"[Test] {{true Label : Count }} : {ground_truth_history}")
    print(f"[Test] {{Model Output Label : Count}} : {predict_history}")
    print(f"[Test] {{True Label : Correct Predict Count}} : {correct_history}")
    print("[Test] Epoch: {}\tCE Loss: {:.6f}\tAcc:{:.1f}% ({}/{})".format(epoch, test_loss, acc, correct, total))

    return test_loss, acc, correct_history, predict_history

def train_step(model, train_loader, criterion, optimizer, epoch, device, log_interval=5):
    model.train()
    train_loss = 0.
    correct = 0
    total = 0

    epoch_size = len(train_loader.dataset)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        if len(targets.size()) == 2:
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets

        correct += predicted.eq(target_labels).sum().item()

        if (batch_idx + 1) % log_interval == 0:
            prog = total/epoch_size
            exact_epoch = epoch + prog - 1
            acc = 100. * correct/total

            if len(targets.shape) >= 2:
                outputs = outputs.cpu().detach()
                output_probs = F.softmax(outputs, dim=1)
                targets = targets.cpu().detach()
                dist = F.mse_loss(output_probs, targets)
                mse_loss = dist.item()
                print(
                        "[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f} MSE (per image in the sampled batch): {.6f}"
                        "\tAccuracy: {:.1f} ({}/{})".format(exact_epoch, batch_idx * len(inputs), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            train_loss/(batch_idx+1), mse_loss, acc, correct, total))
        acc = 100 * correct / total
        return train_loss / (batch_idx+1), acc


def train_model(model, trainset, testset, out_path="../victims/checkpoint/", device=None, batch_size=64, 
                optimizer=None, alpha=1e-3, wd=1e-4, num_workers=10, criterion_train=None, criterion_test=None, 
                lr_step=30, lr_gamma=0.1, epochs=100, log_interval=10, scheduler=None):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")
    model.to(device)
    if not os.path.exists(out_path):
        print("Path {} does not exist. Creating it...".format(out_path))
        os.makedirs(out_path)

    run_id = str(datetime.now())

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if criterion_train is None:
        criterion_train = torch.nn.CrossEntropyLoss(reduction="mean")
    if criterion_test is None:
        criterion_test = torch.nn.CrossEntropyLoss(reduction="mean")

    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    start_epoch=1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, test_acc, test_loss = -1., -1., -1.

    test_correct_history = dict()
    test_predict_history = dict()

    #Logging
    log_path = osp.join(out_path, "train.log.tsv")

    if not osp.exists(log_path):
        with open(log_path, "w") as wf:
            columns = ["run_id", "epoch", "split", "loss", "accuracy", "best_accuracy"]
            wf.write("\t".join(columns) + "\n")

    model_out_path = osp.join(out_path, "checkpoint.pth.tar")
    
    for epoch in range(start_epoch, epochs+1):
        train_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device, log_interval=log_interval)
        scheduler.step(epoch)
        best_train_acc = max(best_train_acc, train_acc)

        #test
        test_loss, test_acc, test_correct_history, test_predict_history = test_step(model, test_loader, criterion_test, device, epoch=epoch)
        best_test_acc = max(best_test_acc, test_acc)

        if test_acc >= best_test_acc:
            state = {
                    "epoch": epoch,
                    "arch": model.__class__,
                    "state_dict": model.state_dict(),
                    "best_acc": test_acc,
                    "optimizer": optimizer.state_dict(),
                    "created_on": str(datetime.now()),
                }
            torch.save(state, model_out_path)

        with open(log_path, "a") as af:
            train_cols = [run_id, epoch, "train", train_loss, train_acc, best_train_acc]
            af.write("\t".join([str(c) for c in train_cols]) + "\n")
            test_correct_cols = [run_id, epoch, "test", "{True Label : Correct Predict Count}", test_correct_history]
            af.write("\t".join([str(c) for c in test_correct_cols]) + "\n")
            test_predict_cols = [run_id, epoch, "test", "{Model Output Label : Count}", test_predict_history]
            af.write("\t".join([str(c) for c in test_predict_cols]) + "\n")
            test_cols = [run_id, epoch, "test", test_loss, test_acc, best_test_acc]
            af.write("\t".join([str(c) for c in test_cols]) + "\n")

    print("Training complete")
    return model
