import torch


def eval(model, data_loader):
    predictions = []
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for data in data_loader:
            input0, input1 = data
            input0, input1 = input0.cuda(), input1.cuda()
            logit = model(input0, input1)

            total_samples += input0.size(0)
            pred = logit.argmax(dim=1)

            predictions.append(pred.tolist())

    return predictions
