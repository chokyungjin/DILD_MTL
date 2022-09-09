import torch

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self. count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_model(name, epoch, model, optimizer,scheduler):
    torch.save({
        'epoch' : epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
        }, name)


def calculate_parameters(model):
    return sum(param.numel() for param in model.parameters())/1000000.0

def pad_collate_fn(batches):
    
    # imgs , masks, labels , patient_id
    X          = [ torch.FloatTensor(batch[0].copy()) for batch in batches ]
    Y          = [ torch.FloatTensor(batch[1].copy()) for batch in batches ]
    labels      = [ batch[2] for batch in batches ]
    id_list    = [ batch[3] for batch in batches ]    
        
    z_shapes = torch.IntTensor([x.shape[-1] for x in X])
    z_mask_shapes = torch.IntTensor([y.shape[-1] for y in Y])
    
    pad_image = []
    pad_label = []
    
    
    for img, label in zip(X, Y):        
        if (z_shapes.max() - img.shape[-1] != 0):
            pad = torch.zeros( (img.shape[0], 
                                img.shape[1], 
                                img.shape[2], 
                                z_shapes.max()-img.shape[3]) )
            pad_image.append(torch.cat([img, pad], dim=-1))

        if (z_mask_shapes.max() - label.shape[-1] != 0):
            pad = torch.zeros( (label.shape[0], 
                                label.shape[1], 
                                label.shape[2], 
                                z_mask_shapes.max()-label.shape[3]) )
            pad_label.append(torch.cat([label, pad], dim=-1))
            
        else :
            pad_image.append(img)
            pad_label.append(label)
            
    batch = dict()
    batch['label']    = labels
    batch['id_list']  = id_list
    batch['image']    = torch.stack(pad_image, dim=0)
    batch['label']    = torch.stack(pad_label, dim=0)
    batch['z_shape']  = z_shapes

    return torch.stack(pad_image, dim=0), torch.stack(pad_label, dim=0), torch.Tensor(labels), id_list
