"""
    Some handy functions for pytroch model training ...
"""
import torch


def get_grad_norm(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.data.view(-1, 1))
    if len(grads) == 0:
        grads.append(torch.FloatTensor([0]))
    grad_norm = torch.norm(torch.cat(grads))
    if grad_norm.is_cuda:
        grad_norm = grad_norm.cpu()
    return grad_norm.item()


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['lr'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=params['lr'],
                                     betas=params['betas'],
                                     weight_decay=params['l2_regularization'],
                                     amsgrad=params['amsgrad'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['lr'],
                                        alpha=params['alpha'],
                                        momentum=params['momentum'],
                                        weight_decay=params['l2_regularization'])
    return optimizer
