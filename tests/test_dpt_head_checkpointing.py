"""Gradient checkpointing in the track head must not change gradients.

``DPTHead.forward`` decodes S frames in chunks and, in training mode, recomputes
each chunk during ``backward()`` instead of retaining it (dpt_head.py).  The
recomputation runs after the chunk loop has finished, so a closure over
``frames_start_idx``/``frames_end_idx`` would replay every chunk with the last
chunk's bounds -- forward values stay bit-identical, nothing raises, and the
gradients are silently wrong for every chunk but the final one.  That is the
shape of the bug ``ce12837`` fixed in the MotionDecoder, which invalidated every
overfit result taken before it.  These tests are the pin against it here.

The oracle is the *single-shot* path (``frames_chunk_size >= S``, the early
return in ``forward``), which has neither chunking nor checkpointing.  Comparing
against it validates the chunk loop and the recomputation together, rather than
only the recomputation.

The head has no dropout, no BatchNorm, no drop-path and no RNG, so ``.train()``
and ``.eval()`` differ *only* in whether chunks are checkpointed -- which is what
makes the exact comparison in
``test_checkpointing_does_not_change_gradients_at_a_fixed_chunk_size`` possible
with no test-only switch on the head.
"""

import pytest
import torch

from arc.models.arc.heads.dpt_head import DPTHead

# Small enough to stay inside the suite's millisecond budget, but a real head:
# four pyramid levels, the production patch_start_idx, and S large enough that
# chunk size 1 produces six separate chunks.
SEQUENCE_LENGTH = 6
PATCH_START_IDX = 1
IMAGE_SIZE = 56
PATCH_GRID = IMAGE_SIZE // 14

# Chunking sums SEQUENCE_LENGTH separate accumulations into each `.grad` where
# the single-shot reference does one batched reduction, so the two agree to
# float32 accumulation order rather than exactly, and the gap grows with S:
# measured max parameter-grad delta 6.0e-07 at S=6 but 3.3e-06 at S=24, against
# gradients of magnitude about 4.9.
#
# CHUNKING_ATOL is the parameter that decides these comparisons -- they pass on
# it alone with rtol at zero, and fail on rtol alone with atol at zero -- so it
# is the one to leave alone.  Headroom is 168x at S=6 but only 30x at S=24: the
# suite's usual atol of 1e-6 leaves just 1.7x margin at S=6 and fails outright
# by S=24, so it would send whoever raises S hunting a gradient bug that does
# not exist.  A mis-bound chunk index, by contrast, moves gradients by 1.3e-02
# even under this mean-reduced loss -- 129x outside the tolerance and 3900x the
# legitimate delta -- so the loosening costs the pin no teeth.  The exact pin
# lives in the fixed-chunk-size test below, where equality is genuinely exact.
CHUNKING_ATOL = 1e-4
CHUNKING_RTOL = 1e-5


def _build_head():
    torch.manual_seed(0)
    return DPTHead(
        dim_in=8,
        output_dim=4,
        features=16,
        out_channels=[8, 8, 8, 8],
        intermediate_layer_idx=[0, 1, 2, 3],
    )


def _build_inputs():
    torch.manual_seed(1)
    tokens = [
        torch.randn(1, SEQUENCE_LENGTH, PATCH_START_IDX + PATCH_GRID * PATCH_GRID, 8)
        for _ in range(4)
    ]
    images = torch.zeros(1, SEQUENCE_LENGTH, 3, IMAGE_SIZE, IMAGE_SIZE)
    return tokens, images


def _gradients(head, training, frames_chunk_size):
    """Loss and gradients for one forward/backward at the given settings."""

    head.train(training)
    head.zero_grad(set_to_none=True)
    tokens, images = _build_inputs()
    tokens = [tensor.clone().requires_grad_(True) for tensor in tokens]

    preds, conf = head(
        tokens,
        images=images,
        patch_start_idx=PATCH_START_IDX,
        frames_chunk_size=frames_chunk_size,
    )
    # Mean-reduced, not summed: under a summed loss the gradient magnitudes are
    # large enough that CHUNKING_ATOL below is inert and rtol alone decides the
    # comparison, which would make the tolerance note next to it a lie.
    loss = preds.square().mean() + conf.square().mean()
    loss.backward()

    parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in head.named_parameters()
        if parameter.grad is not None
    }
    input_gradients = [tensor.grad.detach().clone() for tensor in tokens]
    return loss.detach().clone(), parameter_gradients, input_gradients


def _single_shot(head):
    """The reference: no chunk loop, therefore no checkpointing either."""

    return _gradients(head, training=True, frames_chunk_size=SEQUENCE_LENGTH)


def _assert_gradients_close(actual, expected, label):
    actual_loss, actual_parameters, actual_inputs = actual
    expected_loss, expected_parameters, expected_inputs = expected

    assert set(actual_parameters) == set(expected_parameters)
    assert actual_parameters, "the head must have produced parameter gradients"

    torch.testing.assert_close(
        actual_loss,
        expected_loss,
        atol=CHUNKING_ATOL,
        rtol=CHUNKING_RTOL,
        msg=f"loss differs for {label}",
    )
    for name, gradient in expected_parameters.items():
        torch.testing.assert_close(
            actual_parameters[name],
            gradient,
            atol=CHUNKING_ATOL,
            rtol=CHUNKING_RTOL,
            msg=f"gradient differs for {name} at {label}",
        )
    for index, gradient in enumerate(expected_inputs):
        torch.testing.assert_close(
            actual_inputs[index],
            gradient,
            atol=CHUNKING_ATOL,
            rtol=CHUNKING_RTOL,
            msg=f"gradient differs for input tokens[{index}] at {label}",
        )


def test_checkpointed_chunks_match_the_unchunked_head():
    """The primary pin: six checkpointed chunks against one single-shot pass.

    Verified by reintroducing the bug: closing over the loop bounds leaves the
    forward loss bit-identical (delta 0.0) while moving the worst parameter
    gradient by 1.3e-02, against a legitimate accumulation-order delta of
    6.0e-07 here.
    """

    head = _build_head()
    reference = _single_shot(head)
    chunked = _gradients(head, training=True, frames_chunk_size=1)

    _assert_gradients_close(chunked, reference, "frames_chunk_size=1")


def test_the_loss_is_unchanged_by_chunking_and_checkpointing():
    head = _build_head()
    reference_loss, _, _ = _single_shot(head)
    chunked_loss, _, _ = _gradients(head, training=True, frames_chunk_size=1)

    torch.testing.assert_close(
        chunked_loss,
        reference_loss,
        atol=CHUNKING_ATOL,
        rtol=CHUNKING_RTOL,
    )


@pytest.mark.parametrize("frames_chunk_size", [1, 2, 3])
def test_every_chunk_size_agrees_with_the_unchunked_head(frames_chunk_size):
    """Six, three and two chunks respectively -- all against the same reference."""

    head = _build_head()
    reference = _single_shot(head)
    chunked = _gradients(head, training=True, frames_chunk_size=frames_chunk_size)

    _assert_gradients_close(chunked, reference, f"frames_chunk_size={frames_chunk_size}")


def test_checkpointing_does_not_change_gradients_at_a_fixed_chunk_size():
    """Exactly identical, not merely close -- this is the "unchanged from before" claim.

    Both sides run the same chunk loop over the same bounds in the same order and
    differ only in whether each chunk is recomputed, so the accumulation order is
    identical and the result is bit-exact.  Unlike the comparisons against the
    single-shot path, this cannot drift as S grows, so it is asserted with
    ``torch.equal`` rather than a tolerance.
    """

    head = _build_head()
    _, retained_parameters, retained_inputs = _gradients(
        head, training=False, frames_chunk_size=1
    )
    _, recomputed_parameters, recomputed_inputs = _gradients(
        head, training=True, frames_chunk_size=1
    )

    assert set(recomputed_parameters) == set(retained_parameters)
    for name, gradient in retained_parameters.items():
        assert torch.equal(recomputed_parameters[name], gradient), (
            f"checkpointing changed the gradient for {name}"
        )
    for index, gradient in enumerate(retained_inputs):
        assert torch.equal(recomputed_inputs[index], gradient), (
            f"checkpointing changed the gradient for input tokens[{index}]"
        )


def _retained_bytes(head, training, frames_chunk_size, sequence_length):
    """Bytes of distinct tensor storage held for backward by one forward pass.

    This is the quantity the checkpoint exists to cut, and unlike peak allocation
    it is measurable on CPU, so the change is guarded here rather than only on
    the cluster.  Storages are keyed by identity because the graph saves the same
    tensor at several points.
    """

    head.train(training)
    torch.manual_seed(1)
    tokens = [
        torch.randn(1, sequence_length, PATCH_START_IDX + PATCH_GRID * PATCH_GRID, 8, requires_grad=True)
        for _ in range(4)
    ]
    images = torch.zeros(1, sequence_length, 3, IMAGE_SIZE, IMAGE_SIZE)

    storages = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
        return tensor

    def unpack(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        preds, conf = head(
            tokens,
            images=images,
            patch_start_idx=PATCH_START_IDX,
            frames_chunk_size=frames_chunk_size,
        )
        # Keep the graph alive while the totals are read.
        assert preds.requires_grad and conf.requires_grad

    return sum(storages.values())


def test_checkpointing_cuts_the_tensors_retained_for_backward():
    """The point of the change, measured on CPU at a training-sized sequence."""

    head = _build_head()
    retained = _retained_bytes(head, training=False, frames_chunk_size=1, sequence_length=24)
    recomputed = _retained_bytes(head, training=True, frames_chunk_size=1, sequence_length=24)

    assert recomputed * 4 < retained, (
        f"checkpointing retained {recomputed} bytes against {retained} without it; "
        "expected a multiple-fold reduction"
    )


def test_chunking_alone_retains_as_much_as_the_unchunked_head():
    """Why the checkpoint was needed: chunk size 1 by itself saves nothing.

    Every chunk's activations stay alive until the concatenation, so the loop
    holds the same total as a single pass over all S frames.
    """

    head = _build_head()
    chunked = _retained_bytes(head, training=False, frames_chunk_size=1, sequence_length=24)
    single_shot = _retained_bytes(head, training=False, frames_chunk_size=24, sequence_length=24)

    torch.testing.assert_close(
        float(chunked),
        float(single_shot),
        atol=0.02 * single_shot,
        rtol=0,
    )


def test_peak_cuda_allocation_is_lower_with_checkpointing():
    """Direction only -- the absolute figure is hardware-dependent.

    The CPU retention test above is what guards the change in CI; this exists so
    the claim is also checked in allocator terms wherever a GPU is available.
    """

    if not torch.cuda.is_available():
        pytest.skip("Peak allocation needs a CUDA device")

    head = _build_head().cuda()
    peaks = {}
    for training in (False, True):
        head.train(training)
        head.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        torch.manual_seed(1)
        tokens = [
            torch.randn(
                1, 24, PATCH_START_IDX + PATCH_GRID * PATCH_GRID, 8,
                device="cuda", requires_grad=True,
            )
            for _ in range(4)
        ]
        images = torch.zeros(1, 24, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda")
        preds, conf = head(
            tokens, images=images, patch_start_idx=PATCH_START_IDX, frames_chunk_size=1
        )
        (preds.square().sum() + conf.square().sum()).backward()
        peaks[training] = torch.cuda.max_memory_allocated()

    assert peaks[True] < peaks[False], (
        f"peak allocation was {peaks[True]} with checkpointing against "
        f"{peaks[False]} without it"
    )
