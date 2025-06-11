import torch
import torch.nn as nn
import pytest

from xai.utils import modified_layer
from torchvision.models import vgg16
from xai.lrp_rules import LrpZeroRule, LrpEpsilonRule, LrpGammaRule


torch.manual_seed(0)


def test_modified_layer_transforms_weight_and_bias():
    """modified_layer should apply the transform to both weight and bias."""
    lin = nn.Linear(3, 2, bias=True)
    lin.weight.data = torch.arange(6, dtype=torch.float32).view(2,3)
    lin.bias.data   = torch.arange(2, dtype=torch.float32)

    def plus1(param):
        return param + 1

    new_lin = modified_layer(lin, plus1)

    assert torch.equal(lin.weight, torch.arange(6).view(2,3))
    assert torch.equal(lin.bias,   torch.arange(2))
    assert torch.equal(new_lin.weight, lin.weight + 1)
    assert torch.equal(new_lin.bias,   lin.bias   + 1)


@pytest.fixture(params=[
    (LrpZeroRule, {}),
    (LrpEpsilonRule, {'epsilon':1e-6}),
    (LrpGammaRule, {'gamma':0.3}),
])
def rule_fixture(request):
    """Parameterized fixture for all three LRP rules."""
    RuleCls, kwargs = request.param
    return RuleCls, kwargs


@pytest.fixture
def simple_linear():
    """A simple linear layer for basic forward tests."""
    lin = nn.Linear(4, 2, bias=True)
    return lin


@pytest.fixture
def simple_conv():
    """A simple conv layer for basic forward tests."""
    conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
    conv.weight.data = torch.tensor([[[[1., -2.], [3., -4.]]]])
    return conv


def test_forward_identity_simple_layers(rule_fixture, simple_linear, simple_conv):
    """Each LRP rule should preserve forward activations on simple layers."""
    RuleCls, kwargs = rule_fixture
    # Linear
    x_lin = torch.randn(3,4)
    orig_lin = simple_linear(x_lin)
    wrapped_lin = RuleCls(simple_linear, **kwargs)
    out_lin = wrapped_lin(x_lin)
    assert torch.allclose(out_lin, orig_lin, atol=1e-5)
    # Conv
    x_conv = torch.randn(2,1,2,2)
    orig_conv = simple_conv(x_conv)
    wrapped_conv = RuleCls(simple_conv, **kwargs)
    out_conv = wrapped_conv(x_conv)

    assert torch.allclose(out_conv, orig_conv, atol=1e-5)


@pytest.mark.parametrize("LayerCls,bias", [
    (nn.Linear,   True),
    (nn.Linear,   False),
    (nn.Conv2d,   True),
    (nn.Conv2d,   False),
])
def test_smoke_forward_identity(LayerCls, bias):
    torch.manual_seed(0)
    # build a tiny layer
    if LayerCls is nn.Linear:
        layer = LayerCls(10, 5, bias=bias)
        x = torch.randn(2, 10)
    else:
        layer = LayerCls(3, 4, kernel_size=3, padding=1, bias=bias)
        x = torch.randn(2, 3, 8, 8)

    # pick one of each rule
    for RuleCls, kwargs in [
        (LrpZeroRule,    {}),
        (LrpEpsilonRule, {"epsilon":1e-6}),
        (LrpGammaRule,   {"gamma":0.2}),
    ]:
        orig = layer(x)
        wrapped = RuleCls(layer, **kwargs)
        out   = wrapped(x)
        assert torch.allclose(out, orig, atol=1e-6), (
            f"{RuleCls.__name__} on {LayerCls.__name__} (bias={bias}) "
            "did not preserve forward activations"
        )


if __name__ == "__main__":
    pytest.main()
