import torch

def setup_device():
    """Setup the device for training or inference."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")
    return device

def save_checkpoint(model, optimizer, epoch, filepath):
    """Save model checkpoint."""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer):
    """Load model checkpoint."""
    state = torch.load(filepath)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    return state['epoch']
