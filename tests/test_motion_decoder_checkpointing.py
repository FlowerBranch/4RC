"""Gradient checkpointing in MotionDecoder must not change gradients.

The decoder checkpoints its layer-2 and layer-3 segments in training mode only
(motiondecoder.py), and non-reentrant recomputation runs during ``backward()``,
after the layer loop has finished.  A checkpoint lambda that closed over the
loop variable would therefore recompute every checkpointed segment with the
*last* layer's weights -- forward values stay bit-identical, no error is
raised, and the gradients are silently wrong for the checkpointed layers and
everything upstream of them, including the time-index embedding.  The module
has no dropout, no BatchNorm and zero drop-path, so train-mode and eval-mode
gradients must agree exactly; this is the pin against that regression.
"""

import pytest
import torch

from arc.models.arc.heads.motiondecoder import MotionDecoder


def _gradients(module, tokens, images, training):
    module.train(training)
    module.zero_grad(set_to_none=True)
    inputs = tokens.clone().requires_grad_(True)
    output = module(inputs, images=images, patch_start_idx=2, track_query_idx=0)
    output.square().sum().backward()
    parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }
    return parameter_gradients, inputs.grad.detach().clone()


@pytest.mark.parametrize("use_adaln", [True, False])
def test_training_checkpointing_matches_eval_gradients(use_adaln):
    torch.manual_seed(0)
    decoder = MotionDecoder(
        patch_size=14,
        embed_dim=64,
        depth=4,
        num_heads=4,
        use_adaln=use_adaln,
    )
    tokens = torch.randn(1, 3, 2 + 4, 64)
    images = torch.zeros(1, 3, 3, 28, 28)

    eval_gradients, eval_input_gradient = _gradients(
        decoder, tokens, images, training=False
    )
    train_gradients, train_input_gradient = _gradients(
        decoder, tokens, images, training=True
    )

    assert set(train_gradients) == set(eval_gradients)
    for name, gradient in eval_gradients.items():
        torch.testing.assert_close(
            train_gradients[name],
            gradient,
            atol=1e-6,
            rtol=1e-5,
            msg=f"checkpointed gradient differs for {name}",
        )
    torch.testing.assert_close(
        train_input_gradient,
        eval_input_gradient,
        atol=1e-6,
        rtol=1e-5,
        msg="checkpointed gradient differs for the decoder input",
    )
