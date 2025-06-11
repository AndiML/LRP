import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LRPLinear(nn.Linear):
    """
    A drop-in replacement for nn.Linear that implements the ε-rule
    Layer-wise Relevance Propagation by hijacking autograd’s backward.
    """

    def __init__(self, in_features, out_features, bias=True, eps=1e-6):
        super().__init__(in_features, out_features, bias)
        self.eps = eps

        self.register_forward_hook(self._cache_input)
        self.register_full_backward_hook(self._lrp_backward)

    def _cache_input(self, module, inp, out):
        self._a = inp[0].detach()      # (B, in_features)
        self._z = out.detach()         # pre-activation scores (B, out_features)

    def _lrp_backward(self, module, grad_in, grad_out):
        """
        Receives relevance R_j from upper layer as grad_out[0]
        and returns relevance R_i for the lower layer **instead** of dL/dx.
        """
        R_j = grad_out[0]                             # (B, out_features)
        a   = self._a                                # (B, in_features)
        w   = self.weight                             # (out, in)
        z = torch.einsum('bi,oi->bo', a, w) + self.eps * torch.sign(self._z)
        s = R_j / z
        c = torch.einsum('bi,oi,bo->bi', a, w, s)
        R_i = c                                       
        return (R_i,)
    





class SimpleMLP(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.net = nn.Sequential(
            LRPLinear(28*28, 256, eps=eps, bias=False),  # layer 1
            LRPLinear(256, 10,  eps=eps, bias=False)     # layer 2
        )

    def forward(self, x):
        x = x.flatten(1)
        return self.net(x)



def lrp_epsilon(model, x, target=None):
    """
    Computes ε-LRP relevances for a single sample (or batch).
    x must require grad so that autograd can store R in x.grad.
    """
    x = x.clone().requires_grad_(True)
    out = model(x)                         # forward pass

    # focus relevance on the desired logit
    if target is None:
        target = out.argmax(dim=1)
    score = out[range(out.shape[0]), target].sum()

    model.zero_grad(set_to_none=True)
    score.backward()                       # triggers our LRP hooks
    return x.grad                          # R for each input pixel



BATCH = 32
RTOL, ATOL = 1e-3, 1e-6          # numeric slack for allclose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_mnist_loader(batch_size=BATCH):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)


def test_lrp_epsilon_conservation():
    """
    For every image in the batch the sum of input relevances must equal
    the score that was back-propagated AND the sums must agree over the whole
    batch.
    """

    model = SimpleMLP(eps=1e-6).to(device).eval()
    images, _ = next(iter(_get_mnist_loader()))
    images = images.to(device)

    # 3) compute relevances
    R = lrp_epsilon(model, images)              # (B, 1, 28, 28)
    R_flat = R.flatten(1)                       # (B, 784)


    # 4) forward pass (no grads) to get the same scores LRP used
    with torch.no_grad():
        logits = model(images)
        chosen  = logits[range(len(images)), logits.argmax(1)]   # (B,)

    print(R_flat.sum())
    print(chosen.sum())
    # 5) assertions -----------------------------------------------------------
    # per-sample
    assert torch.allclose(R_flat.sum(dim=1), chosen,
                          rtol=RTOL, atol=ATOL), \
        "Relevance is not conserved for at least one sample"

    # whole batch
    assert torch.allclose(R_flat.sum(), chosen.sum(),
                          rtol=RTOL, atol=ATOL), \
        "Batch-wise relevance sum differs from propagated score"


if __name__ == "__main__":
    # Quick manual run:  python test_lrp_mnist.py
    test_lrp_epsilon_conservation()
    print("✓ ε-LRP conservation test passed")
