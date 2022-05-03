import torch


def test_epoch(args, model, device, data_loader, criterion, LOGGER=None):
    model.eval()
    test_loss = 0
    correct = 0

    data_len = len(data_loader.dataset)

    with torch.no_grad():
        for step, (data, target) in enumerate(data_loader):
            # target = torch.zeros(len(target), target.max() + 1).scatter_(1, target.unsqueeze(1), 1.)  # one-hot encoding
            target = target.to(device)
            output = model(data.to(device))
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1)[1]  # get the index of the max log-probability
            # pred = torch.round(output)
            correct += pred.eq(target).sum().item()

    try:
        accuracy = 100. * correct / data_len
    except:
        accuracy = 100. * correct / (data_len + 1e-8)

    try:
        test_loss /= (step / args.num_workers)
    except:
        test_loss /= (19 / args.num_workers)
    LOGGER.info(f'Average loss: {test_loss:.8f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)')

    return test_loss, accuracy


def test_fn(args, model, device, test_loader, criterion, LOGGER=None):
    # start_time = time.time()
    LOGGER.info("\n=====Test=====")

    loss, accuracy = test_epoch(args, model, device, test_loader, criterion, LOGGER)
    # end_time = time.time()
    # LOGGER.info(f'Evaluation Time : {end_time - start_time}')
    return loss, accuracy
