"""Local-attention checkpointing in the encoder must not change gradients.

``process_attention`` (vision_transformer.py) checkpoints the global-attention
blocks in training mode unconditionally.  ``checkpoint_local_attention`` extends
that to the local-attention blocks -- 26 of the 40 at ``alt_start=13``, and the
bulk of what the encoder retains for backward.  This file pins both halves of the
claim: that turning it on changes no gradient, and that it actually cuts the
retained tensors.

The encoder has no dropout, no BatchNorm and ``drop_path_rate=0.0`` (never
overridden -- ``DinoV2`` does not pass it), so a block's output does not depend
on whether it was recomputed, and the flag is the *only* thing separating the two
sides of every comparison below.  The checkpoint lambda takes its captured values
as arguments rather than closing over them, which is what makes that true during
``backward()`` as well; ``ce12837`` is the bug where that was not so in
``MotionDecoder``, and the forward stayed bit-identical while the gradients went
silently wrong.

**On the exactness of the gradient tests.**  They assert ``torch.equal``, not a
tolerance, and that is measured rather than assumed: at the routing used here all
675 parameter gradients, the input gradient and the loss come back bit-identical.
This holds because CPU fp32 eager execution runs the recomputed forward through
exactly the kernels the retained one used, in the same order.  It is NOT a claim
about training runs: under CUDA with bf16 autocast, recomputation can select
different attention and matmul kernels, so the ON path is a memory/compute trade
that reorders floating-point work and is not guaranteed bit-identical there.  The
invariant that does carry to the cluster is the other one -- default OFF changes
nothing at all, which ``test_the_flag_defaults_off`` and
``test_off_retains_exactly_what_it_retained_before`` pin.
"""

from types import SimpleNamespace

import pytest
import torch

from arc.models.arc.arc import Arc
from arc.models.arc.dinov2.vision_transformer import vit_small

# Small enough for the suite's budget, but the real routing: alt_start splits the
# blocks into a leading all-local run plus an alternating tail, so both branches
# of process_attention are exercised and the flag has local blocks to act on.
DEPTH = 8
ALT_START = 3
IMAGE_SIZE = 70
PATCH_SIZE = 14
BATCH = 1
FRAMES = 3
OUT_LAYERS = [DEPTH - 1]


def _local_block_count(depth=DEPTH, alt_start=ALT_START):
    """The blocks the flag acts on: everything process_attention routes local."""

    return depth - len([i for i in range(alt_start, depth) if i % 2 == 1])


def _build_encoder():
    torch.manual_seed(0)
    return vit_small(
        img_size=518,
        patch_size=PATCH_SIZE,
        ffn_layer="mlp",
        alt_start=ALT_START,
        qknorm_start=ALT_START,
        rope_start=ALT_START,
        cat_token=True,
        has_time_token=True,
        max_time_indices=32,
        depth=DEPTH,
    )


def _inputs():
    torch.manual_seed(1)
    images = torch.randn(BATCH, FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
    time_indices = torch.arange(FRAMES).unsqueeze(0).expand(BATCH, FRAMES).contiguous()
    return images, time_indices


def _gradients(encoder, local_checkpointing):
    """Loss and gradients for one forward/backward at the given flag value."""

    encoder.train(True)
    encoder.checkpoint_local_attention = local_checkpointing
    encoder.zero_grad(set_to_none=True)

    images, time_indices = _inputs()
    images = images.clone().requires_grad_(True)

    output, _aux = encoder.get_intermediate_layers(
        images, OUT_LAYERS, time_indices=time_indices
    )
    # Every tensor the encoder hands back -- patch tokens, camera token and, under
    # has_time_token, the time token -- so the comparison covers the whole output
    # rather than whichever member happens to be first.
    loss = sum(tensor.square().mean() for entry in output for tensor in entry)
    loss.backward()

    parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in encoder.named_parameters()
        if parameter.grad is not None
    }
    return loss.detach().clone(), parameter_gradients, images.grad.detach().clone()


def _retained_bytes(encoder, local_checkpointing):
    """Bytes of distinct tensor storage held for backward by one forward pass.

    This is the quantity the flag exists to cut, and unlike peak allocation it is
    measurable on CPU, so the change is guarded here rather than only on the
    cluster.  Storages are keyed by identity because the graph saves the same
    tensor at several points.

    Keying by `data_ptr()` makes the total approximate: the allocator reuses freed
    pointers, so two measurements of the same encoder can differ by a few bytes in
    49 MB.  Fine for the multiple-fold comparison this exists for; not a basis for
    asserting equality.
    """

    encoder.train(True)
    encoder.checkpoint_local_attention = local_checkpointing

    images, time_indices = _inputs()
    images = images.requires_grad_(True)
    storages = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
        return tensor

    def unpack(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output, _aux = encoder.get_intermediate_layers(
            images, OUT_LAYERS, time_indices=time_indices
        )
        # Keep the graph alive while the totals are read.
        assert all(tensor.requires_grad for entry in output for tensor in entry)
        return sum(storages.values())


def test_the_flag_defaults_off():
    """The invariant the whole change rests on: nothing moves unless asked."""

    assert _build_encoder().checkpoint_local_attention is False


def test_the_accessor_reaches_the_encoder_the_model_actually_runs():
    """`Arc.set_encoder_local_checkpointing` must land on `backbone.pretrained`.

    This is the failure the whole change was nearly built on: there are two
    `vision_transformer.py` files, and the orphaned `arc/models/arc/layers` copy
    carries a `use_checkpoint` attribute that nothing reads.  An accessor writing
    to the wrong handle would raise nothing, change nothing, and leave the flag
    reported in run_summary.json while the run trained exactly as before.

    Called unbound against a stand-in rather than on a real Arc, which cannot be
    built on the meta device (the drop-path linspace calls `.item()`) and costs
    1.1B parameters on CPU.  What is under test is the attribute path, and the
    stand-in has the same shape as the real one.
    """

    encoder = _build_encoder()
    model = SimpleNamespace(backbone=SimpleNamespace(pretrained=encoder))

    Arc.set_encoder_local_checkpointing(model, True)
    assert encoder.checkpoint_local_attention is True

    Arc.set_encoder_local_checkpointing(model, False)
    assert encoder.checkpoint_local_attention is False

    # Coerced, so argparse's value and a bare truthy one behave identically.
    Arc.set_encoder_local_checkpointing(model, 1)
    assert encoder.checkpoint_local_attention is True


def test_the_routing_leaves_local_blocks_for_the_flag_to_act_on():
    """Guards the fixture, not the code: a depth/alt_start pair that happened to
    route everything global would make every comparison below vacuously pass."""

    assert _local_block_count() > 0
    assert _local_block_count() < DEPTH


def test_checkpointing_local_attention_does_not_change_gradients():
    """The primary pin. Exact, not close -- see the module docstring."""

    encoder = _build_encoder()
    retained_loss, retained_parameters, retained_input = _gradients(encoder, False)
    recomputed_loss, recomputed_parameters, recomputed_input = _gradients(encoder, True)

    assert torch.equal(recomputed_loss, retained_loss), (
        "checkpointing changed the forward loss, which it cannot do"
    )
    assert set(recomputed_parameters) == set(retained_parameters)
    assert retained_parameters, "the encoder must have produced parameter gradients"
    for name, gradient in retained_parameters.items():
        assert torch.equal(recomputed_parameters[name], gradient), (
            f"checkpointing changed the gradient for {name}"
        )
    assert torch.equal(recomputed_input, retained_input), (
        "checkpointing changed the gradient for the encoder input"
    )


def test_checkpointing_cuts_the_tensors_retained_for_backward():
    """The point of the change, measured on CPU."""

    encoder = _build_encoder()
    retained = _retained_bytes(encoder, False)
    recomputed = _retained_bytes(encoder, True)

    assert recomputed * 4 < retained, (
        f"local checkpointing retained {recomputed} bytes against {retained} "
        "without it; expected a multiple-fold reduction"
    )


def test_off_is_indistinguishable_from_the_shipped_behaviour():
    """Default OFF must change nothing -- the invariant the cluster runs rest on.

    The flag is only ever read as a disjunct in process_attention's gate, so with
    it False the expression is literally the one that shipped.  Pinned on the
    gradients, which are exactly reproducible, rather than on retained bytes,
    which are not: `_retained_bytes` keys storages by `data_ptr()`, and a freed
    pointer can be reused between calls, so the total drifts by a few bytes in
    49 MB even for one encoder measured twice.  That noise is ~0.01% and nowhere
    near the multiple-fold gap the reduction test asserts, but it is far too
    large a foundation for an equality claim -- do not tighten this back.
    """

    untouched = _build_encoder()
    explicit = _build_encoder()
    explicit.checkpoint_local_attention = False

    untouched_loss, untouched_parameters, untouched_input = _gradients(untouched, False)
    explicit_loss, explicit_parameters, explicit_input = _gradients(explicit, False)

    assert torch.equal(explicit_loss, untouched_loss)
    assert set(explicit_parameters) == set(untouched_parameters)
    for name, gradient in untouched_parameters.items():
        assert torch.equal(explicit_parameters[name], gradient), (
            f"setting the flag False changed the gradient for {name}"
        )
    assert torch.equal(explicit_input, untouched_input)


def test_the_flag_is_inert_in_eval_mode():
    """The gate requires self.training, so held-out eval is untouched either way.

    evaluate_held_out runs the model in eval mode; if the flag leaked into that
    path it would recompute for no benefit, since eval builds no graph to save.
    """

    encoder = _build_encoder()
    images, time_indices = _inputs()

    outputs = []
    for local_checkpointing in (False, True):
        encoder.eval()
        encoder.checkpoint_local_attention = local_checkpointing
        with torch.no_grad():
            output, _aux = encoder.get_intermediate_layers(
                images, OUT_LAYERS, time_indices=time_indices
            )
        outputs.append(output[0][0])

    assert torch.equal(outputs[0], outputs[1])


def test_peak_cuda_allocation_is_lower_with_checkpointing():
    """Direction only -- the absolute figure is hardware-dependent.

    The CPU retention test above is what guards the change in CI; this exists so
    the claim is also checked in allocator terms wherever a GPU is available.
    """

    if not torch.cuda.is_available():
        pytest.skip("Peak allocation needs a CUDA device")

    encoder = _build_encoder().cuda()
    images, time_indices = _inputs()
    images, time_indices = images.cuda(), time_indices.cuda()

    peaks = {}
    for local_checkpointing in (False, True):
        encoder.train(True)
        encoder.checkpoint_local_attention = local_checkpointing
        encoder.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        output, _aux = encoder.get_intermediate_layers(
            images, OUT_LAYERS, time_indices=time_indices
        )
        sum(
            tensor.square().mean() for entry in output for tensor in entry
        ).backward()
        peaks[local_checkpointing] = torch.cuda.max_memory_allocated()

    assert peaks[True] < peaks[False], (
        f"peak allocation was {peaks[True]} with local checkpointing against "
        f"{peaks[False]} without it"
    )
