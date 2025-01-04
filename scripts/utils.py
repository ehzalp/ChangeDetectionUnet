from imports import *

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def jaccard_index(pred, target):
    intersection = torch.sum(pred * target, dim=(2, 3))
    union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) - intersection
    jaccard_per_image = torch.where(union == 0, torch.tensor(1.0), (intersection + 1e-10) / (union + 1e-10))
    return torch.mean(jaccard_per_image, dim=1)

def calc_loss(pred, target, metrics, bce_weight=0.8):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    jaccard_per_image = jaccard_index((pred > 0.5).float(), target)
    jaccard = torch.mean(jaccard_per_image)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    metrics['bce'] += bce.data.to("cuda:0") * target.size(0)
    metrics['dice'] += dice.data.to("cuda:0") * target.size(0)
    metrics['jaccard'] += jaccard.to("cuda:0") * target.size(0)
    metrics['loss'] += loss.data.to("cuda:0") * target.size(0)
    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    if phase != 'test':
        outputs.append("Jaccard Index: {:4f}".format(metrics['jaccard'] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, dataloaders, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs_combined, labels in dataloaders[phase]:
                inputs_combined = inputs_combined.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs_combined)
                    loss = calc_loss(outputs, labels, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs_combined.size(0)

            if phase == 'train':
                scheduler.step()

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model