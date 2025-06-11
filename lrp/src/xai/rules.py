import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets

_activation_store = {}
_weight_store     = {}
_relevance_store  = {}

class RelevanceProgator:
    pass


def _linear_forward_hook(module, inputs, outputs):
    x = inputs[0].detach()            
    W = module.weight.detach()
    b = module.bias.detach() if (module.bias is not None) else None

    _activation_store[module] = x
    _weight_store[module]     = (W, b)

def register_forward_hooks(model):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.register_forward_hook(_linear_forward_hook)

def _linear_backward_hook(module, grad_input, grad_output):
    # upstream relevance R^(l)
    R_l = grad_output[0].detach()
    # Retrieve activation
    x = _activation_store[module]      
    W, b = _weight_store[module]
    eps   = 1e-6
    gamma = 0.25 
    W_pos = torch.clamp(W, min=0.0)
    W_rho = W_pos + gamma * W_pos
    z = x.matmul(W_rho.t()) + eps
    s = R_l / z
    c = s.matmul(W_rho)
    R_prev = x * c
    _relevance_store[module] = R_prev.detach()

    return (R_prev,)


def register_backward_hooks(model):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.register_full_backward_hook(_linear_backward_hook)


# ─── Simple MLP on MNIST ─────────────────────────────────────────────────────
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1   = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3   = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)       # flatten 28×28 → 784
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        out = self.fc3(h2)
        return out


# ─── Main script ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = SimpleMLP()
    register_forward_hooks(model)
    register_backward_hooks(model)

    # Load one MNIST example
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./datasets',
                                   train=True,
                                   download=True,
                                   transform=transform)
    img, label = train_dataset[0]     # img shape = [1,28,28]
    x = img.unsqueeze(0)              # [1,1,28,28]

    logits = model(x)                 # caches a^(l-1) & W in our stores

    # 2) Build one-hot for the predicted class
    scores   = logits                 # shape [1,10]
    pred_idx = scores.argmax(dim=1)   # e.g. tensor([7])
    one_hot  = torch.nn.functional.one_hot(pred_idx, 10)

    # 3) Backward => triggers our LRP hook
    model.zero_grad()
    R_out = (scores * one_hot).sum()
    R_out.backward( )

    # 4) Extract the first‐layer relevance (784 dims) and reshape to [1,1,28,28]
    R_first = _relevance_store[ model.fc1 ]    # shape [1,784]
    R_pixel = R_first.view(1, 1, 28, 28)       # [1,1,28,28]
    exit()
    # 5) Pixel‐flipping experiment
    flat_relevance = R_first.flatten()           # [784]
    _, sorted_idx = torch.sort(flat_relevance, descending=True)

    orig_flat = x.view(1, 784).clone()           # [1,784]
    num_pixels = 784
    scores_after_flip = torch.zeros(num_pixels + 1)

    # k = 0 (no flips)
    with torch.no_grad():
        orig_logits = model(x)
        scores_after_flip[0] = orig_logits[0, pred_idx].item()

    for k in range(1, num_pixels + 1):
        flipped_flat = orig_flat.clone()
        pix_to_zero  = sorted_idx[k - 1].item()
        flipped_flat[0, pix_to_zero] = 0.0
        flipped_img = flipped_flat.view(1, 1, 28, 28)

        with torch.no_grad():
            logits_k                = model(flipped_img)
            scores_after_flip[k]    = logits_k[0, pred_idx].item()

    # 6) Plot the flipping curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 4))
    plt.plot(range(num_pixels + 1), scores_after_flip.numpy(), linewidth=2)
    plt.xlabel("Number of pixels zeroed (descending relevance)")
    plt.ylabel(f"Score for class {pred_idx.item()}")
    plt.title("Pixel‐Flipping Curve")
    plt.grid(True)
    plt.show()
