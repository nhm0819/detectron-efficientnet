import os
import torch
import time

from efficientnet.clf_test_function import test_epoch


def train_fn(args, model, device, train_loader, val_loader, writer, LOGGER, scaler, criterion, optimizer, scheduler, stage=1):

    best_accuracy = 0

    # training for epochs
    for epoch in range(args.epoch_per_stage*(stage-1) + 1, args.epoch_per_stage*stage + 1):
        model.train()
        train_loss, train_accuracy, train_lr = train_epoch(epoch, args, model, device, train_loader, criterion,
                                                           optimizer, scheduler, scaler, LOGGER)

        writer.add_scalars('Loss', {'train': train_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_accuracy}, epoch)
        writer.add_scalars('LR', {'train': train_lr}, epoch)

        if epoch % args.test_per_epochs == 0:
            LOGGER.info(f'\n========== Validation ==========')
            val_loss, val_accuracy = test_epoch(args, model, device, val_loader, criterion, LOGGER)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(),
                           os.path.join(args.eff_output_dir, f'efficientnet_b4_acc_{best_accuracy:.2f}.pt'))
                LOGGER.info(f'Save the best acc model, loss = {val_loss:.2f}, acc = {best_accuracy:.2f}')

            writer.add_scalars('Loss', {'val': val_loss}, epoch)
            writer.add_scalars('Accuracy', {'val': val_accuracy}, epoch)



def train_epoch(epoch, args, model, device, train_loader, criterion, optimizer, scheduler, scaler, LOGGER=None):
    train_loss = []
    correct = 0

    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.to(device)

        optimizer.zero_grad()

        # output
        output = model(data.to(device))
        loss = criterion(output, target)

        # get loss
        train_loss.append(loss.item())

        # get preds (for accuracy)
        pred = output.max(1)[1]
        correct += pred.eq(target).sum().item()

        # loss.backward()
        scaler.scale(loss).backward()

        # # to apply scheduler
        scaler.unscale_(optimizer)
        scale_before = scaler.get_scale()

        # optimizer.step()
        scaler.step(optimizer)

        # scaler update
        scaler.update()

        # scheduler.step()
        scale_after = scaler.get_scale()
        if scale_before <= scale_after:
            scheduler.step()

        if batch_idx % args.log_interval == 0:
            LOGGER.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.8f}')


    mean_loss = sum(train_loss) / (len(train_loss) + 1e-7)
    accuracy = 100. * correct / len(train_loader.dataset)
    LR = optimizer.param_groups[0]['lr']

    end_time = time.time()
    LOGGER.info(f'Train Mean Loss : {mean_loss:.8f},\tTrain Accuracy: {accuracy:.2f},'
                f'\tTime : {end_time - start_time:.2f} sec\t LR : {LR:.8f}\t')

    return mean_loss, accuracy, LR
