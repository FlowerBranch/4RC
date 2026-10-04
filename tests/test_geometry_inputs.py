"""The --depth_input / --camera_input capability, end to end.

Encoder-level: the injections execute yet change nothing at zero init (the
**kwargs trap cannot pass), depth tokens land on their own patches, camera
vectors ride the reference reorder, all four arms are constructible. Loader
level: the depth channels are gathered (never interpolated), sanitised and
invalidated beyond --kubric_max_depth; the camera vector is the model-grid
camera-to-world encoding plus the principal point. Guard level: the gradient
guards under single-flag arms, the CUDA move, the keyword-only surfaces, and
the Arc-to-backbone route pin.
"""

import ast
import inspect
import math
import textwrap

import numpy as np
import pytest
import torch
from torch import nn

import arc.training.dumped_kubric as dumped_kubric
import arc.training.runtime as runtime
from arc.models.arc.arc import Arc
from arc.models.arc.dinov2.vision_transformer import DinoVisionTransformer
from arc.models.arc.utils.transform import pose_encoding_to_extri_intri
from scene_fixtures import fixture_scene
from test_time_indexing import (
    _PassThroughTimeTransformer,
    _arc_shell,
    _FakeBackbone,
    _FakeCameraDecoder,
    _FakeMotionDecoder,
    _FakeReconstructionHead,
    _FakeTrackHead,
    _run_pass_through,
)


def _small_encoder(**overrides):
    """test_time_indexing's real-encoder shape: ONE 14x14 patch per view.

    Deliberately blind to token order -- at one patch any permutation of the
    depth tokens is unobservable -- which is why the mapping test below uses
    the 2x2 grid encoder instead.
    """

    config = dict(
        img_size=14,
        patch_size=14,
        embed_dim=8,
        depth=3,
        num_heads=2,
        mlp_ratio=2,
        alt_start=2,
        has_time_token=True,
        cat_token=False,
        max_time_indices=8,
    )
    config.update(overrides)
    return DinoVisionTransformer(**config)


def _grid_encoder():
    """The img-28 probe fixture: a 2x2 patch grid, where token order shows."""

    return DinoVisionTransformer(
        img_size=28,
        patch_size=14,
        embed_dim=8,
        depth=6,
        num_heads=2,
        ffn_layer="mlp",
        alt_start=3,
        qknorm_start=3,
        rope_start=3,
        cat_token=True,
        has_time_token=True,
        max_time_indices=4,
    )


def _quaternion_xyzw(matrix):
    """Independent rotation-matrix -> xyzw quaternion, test-side (Shepperd)."""

    m = np.asarray(matrix, dtype=np.float64)
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0
    x = np.copysign(x, m[2, 1] - m[1, 2])
    y = np.copysign(y, m[0, 2] - m[2, 0])
    z = np.copysign(z, m[1, 0] - m[0, 1])
    return np.array([x, y, z, w])


def test_depth_branch_executes_and_zero_init_is_inert():
    """The **kwargs-trap killer.

    An argument silently swallowed by prepare_tokens_with_masks's ignored
    **kwargs is indistinguishable from working code under zero init -- both
    leave the output byte-identical. So beyond inertness this asserts the
    branch EXECUTED: a filled conv must move the output, and the zero conv
    must still receive a gradient.
    """

    torch.manual_seed(0)
    encoder = _small_encoder()
    images = torch.randn(1, 4, 3, 14, 14)
    depth = torch.rand(1, 4, 2, 14, 14) + 0.5

    base = _run_pass_through(encoder, images)
    inert = _run_pass_through(encoder, images, depth_maps=depth)
    assert torch.equal(inert, base)

    with torch.no_grad():
        encoder.depth_patch_embed.weight.fill_(1.0)
    hot = _run_pass_through(encoder, images, depth_maps=depth)
    assert not torch.equal(hot, base)

    with torch.no_grad():
        encoder.depth_patch_embed.weight.zero_()
    out = _run_pass_through(encoder, images, depth_maps=depth)
    out.sum().backward()
    grad = encoder.depth_patch_embed.weight.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0


def test_depth_tokens_land_on_their_own_patch():
    """The mapping pin the one-patch geometry above cannot carry.

    At img 14 a transposed flatten is bit-identical; at the 2x2 grid it is
    not, so a permuted flatten passes every other test in this file and fails
    here. Depth nonzero in exactly ONE off-diagonal grid cell must move
    exactly that cell's token. The comparison runs on prepare_tokens_with_masks
    itself so attention cannot smear the delta across tokens first.
    """

    torch.manual_seed(0)
    encoder = _grid_encoder()
    images = torch.randn(1, 4, 3, 28, 28)
    depth = torch.zeros(1, 4, 2, 28, 28)
    # Grid cell (row 1, column 0): token 1 + row*2 + column = 3 under the
    # row-major flatten; a transposed flatten would land it on token 2.
    depth[:, :, :, 14:, :14] = 1.0
    with torch.no_grad():
        encoder.depth_patch_embed.weight.fill_(1.0)

    base = encoder.prepare_tokens_with_masks(images)
    with_depth = encoder.prepare_tokens_with_masks(images, depth_maps=depth)

    changed = (with_depth - base).abs().sum(-1)[0]
    for view in range(4):
        assert torch.nonzero(changed[view]).flatten().tolist() == [3]


def test_injection_construction_is_rng_guarded_in_the_real_constructor():
    """The half of flag-off bit-identity no output comparison can see.

    Constructing Conv2d/Linear draws from the global RNG before the zeros
    overwrite, which would shift every later randn() against pre-injection
    code. A naive "state unchanged" assert on the whole constructor would be
    wrong -- it legitimately draws for patch_embed, the tokens and the blocks
    -- so the pin is LEXICAL, the repo's inspected-rather-than-executed
    precedent: both injection assignments in DinoVisionTransformer.__init__
    must sit inside a fork_rng With block. Deleting either guard fails this
    and nothing else in the suite (the one-off pristine-tree probe leaves no
    artifact). The canary then proves fork_rng actually isolates the draws
    the guard exists to contain.
    """

    source = textwrap.dedent(inspect.getsource(DinoVisionTransformer.__init__))
    guarded = {}

    def visit(node, inside_fork):
        if isinstance(node, ast.With):
            inside_fork = inside_fork or any(
                isinstance(item.context_expr, ast.Call)
                and getattr(item.context_expr.func, "attr", None) == "fork_rng"
                for item in node.items
            )
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr in ("depth_patch_embed", "camera_proj")
                ):
                    guarded[target.attr] = inside_fork
        for child in ast.iter_child_nodes(node):
            visit(child, inside_fork)

    visit(ast.parse(source), False)
    assert guarded == {"depth_patch_embed": True, "camera_proj": True}

    torch.manual_seed(0)
    state = torch.random.get_rng_state()
    with torch.random.fork_rng(devices=[]):
        nn.Conv2d(2, 8, kernel_size=14, stride=14)
        nn.Linear(11, 8)
    assert torch.equal(torch.random.get_rng_state(), state)

    nn.Conv2d(2, 8, kernel_size=14, stride=14)
    nn.Linear(11, 8)
    assert not torch.equal(torch.random.get_rng_state(), state)


def test_camera_vector_zero_init_is_inert_and_the_converse_holds():
    """On the REAL encoder, whose camera token is randn and nonzero.

    That is load-bearing: on a zeros-token stub, `camera_proj(v)` (replace)
    is indistinguishable from `cam_token + camera_proj(v)` (add), and a
    replace discards the pretrained token cam_dec reads. Here a replace
    zeroes the randn token and fails the first equality.
    """

    torch.manual_seed(0)
    encoder = _small_encoder()
    images = torch.randn(1, 4, 3, 14, 14)
    vectors = torch.randn(1, 4, 11)

    base = _run_pass_through(encoder, images)
    inert = _run_pass_through(encoder, images, camera_vectors=vectors)
    assert torch.equal(inert, base)

    with torch.no_grad():
        encoder.camera_proj.weight.fill_(1.0)
    hot = _run_pass_through(encoder, images, camera_vectors=vectors)
    assert not torch.equal(hot, base)

    # The converse control: with the projection filled but no vectors passed,
    # the output is still today's -- the first equality was not "never read".
    filled_none = _run_pass_through(encoder, images)
    assert torch.equal(filled_none, base)


def test_camera_vectors_are_refused_without_a_camera_token():
    """No camera token means nowhere to add: accepting the vectors would
    validate them and then silently drop them, which under zero init is
    indistinguishable from working code."""

    torch.manual_seed(0)
    encoder = _small_encoder(alt_start=-1, has_time_token=False)
    images = torch.randn(1, 4, 3, 14, 14)

    with pytest.raises(ValueError, match="no camera token"):
        _run_pass_through(encoder, images, camera_vectors=torch.randn(1, 4, 11))


def test_camera_vectors_ride_the_reference_reorder():
    """Landmine 6, at FOUR views -- the reorder is guarded by
    THRESH_FOR_REF_SELECTION == 3, so a two-view test can never catch an
    un-permuted camera vector pairing camera k's pose with camera j's tokens.
    """

    model = _PassThroughTimeTransformer()
    # Zero by construction; the assert doubles as an init pin, and a random
    # bias would fail the equality below for patched and unpatched code alike.
    assert torch.count_nonzero(model.camera_proj.bias) == 0
    with torch.no_grad():
        model.camera_proj.weight.copy_(torch.eye(model.embed_dim, 11))

    images = torch.zeros(1, 4, 3, 2, 2)
    vectors = torch.zeros(1, 4, 11)
    for view in range(4):
        vectors[0, view, 0] = view + 1

    output = _run_pass_through(model, images, camera_vectors=vectors)

    # The stub's camera_token is zeros and the taps come back in ORIGINAL view
    # order (the reorder test above this file's imports pins that), so each
    # view's slot-0 token must carry its own vector's first entry. Without the
    # reorder line, view s receives another camera's vector and this fails.
    for view in range(4):
        assert float(output[0, view, 0, 0]) == pytest.approx(view + 1)


def test_all_four_arms_are_constructible_and_separate():
    """One encoder, four arms: equal at zero init, separable once filled."""

    torch.manual_seed(0)
    encoder = _small_encoder()
    images = torch.randn(1, 4, 3, 14, 14)
    depth = torch.rand(1, 4, 2, 14, 14) + 0.5
    vectors = torch.randn(1, 4, 11)

    def run(with_depth, with_camera):
        return _run_pass_through(
            encoder,
            images,
            depth_maps=depth if with_depth else None,
            camera_vectors=vectors if with_camera else None,
        )

    arms = {(d, c): run(d, c) for d in (False, True) for c in (False, True)}
    for key, value in arms.items():
        assert torch.equal(value, arms[(False, False)]), key

    with torch.no_grad():
        encoder.depth_patch_embed.weight.fill_(0.5)
        encoder.camera_proj.weight.fill_(0.5)
    hot = {(d, c): run(d, c) for d in (False, True) for c in (False, True)}
    base = hot[(False, False)]
    assert torch.equal(base, arms[(False, False)])
    assert not torch.equal(hot[(True, False)], base)
    assert not torch.equal(hot[(False, True)], base)
    assert not torch.equal(hot[(True, False)], hot[(False, True)])
    assert not torch.equal(hot[(True, True)], hot[(True, False)])
    assert not torch.equal(hot[(True, True)], hot[(False, True)])


def test_geometry_metadata_is_all_or_none_and_validated():
    """The time_index contract, clause for clause, on both new keys."""

    model = _arc_shell()
    image = torch.zeros(1, 3, 2, 2)
    good_depth = torch.zeros(1, 2, 2, 2)
    good_camera = torch.zeros(1, 11)

    with pytest.raises(ValueError, match="every view"):
        model._preprocess_input(
            [{"img": image, "depth_map": good_depth}, {"img": image}]
        )
    with pytest.raises(ValueError, match="every view"):
        model._preprocess_input(
            [{"img": image, "camera_vector": good_camera}, {"img": image}]
        )

    with pytest.raises(ValueError, match=r"\(2, 2, 2\) or \(1, 2, 2, 2\)"):
        model._preprocess_input(
            [{"img": image, "depth_map": torch.zeros(1, 3, 2, 2)} for _ in range(2)]
        )
    with pytest.raises(ValueError, match=r"\(11,\) or \(1, 11\)"):
        model._preprocess_input(
            [{"img": image, "camera_vector": torch.zeros(1, 7)} for _ in range(2)]
        )

    with pytest.raises(ValueError, match="finite"):
        model._preprocess_input(
            [
                {"img": image, "depth_map": torch.full((1, 2, 2, 2), float("inf"))}
                for _ in range(2)
            ]
        )
    with pytest.raises(ValueError, match="finite"):
        model._preprocess_input(
            [
                {"img": image, "camera_vector": torch.full((1, 11), float("nan"))}
                for _ in range(2)
            ]
        )

    _, _, time_indices, depth_maps, camera_vectors = model._preprocess_input(
        [{"img": image} for _ in range(2)]
    )
    assert time_indices is None and depth_maps is None and camera_vectors is None

    _, _, _, depth_maps, camera_vectors = model._preprocess_input(
        [
            {"img": image, "depth_map": good_depth, "camera_vector": good_camera}
            for _ in range(3)
        ]
    )
    assert depth_maps.shape == (1, 3, 2, 2, 2)
    assert camera_vectors.shape == (1, 3, 11)


def test_view_key_constants_cannot_drift():
    """The loader duplicates the model's key spellings to stay import-light;
    this is the drift pin that duplication is licensed by."""

    assert Arc.DEPTH_KEY == dumped_kubric.DEPTH_INPUT_KEY
    assert Arc.CAMERA_VECTOR_KEY == dumped_kubric.CAMERA_VECTOR_KEY


def test_depth_input_is_gathered_not_interpolated():
    """Landmine 11: a depth discontinuity must never be blended into a depth
    that exists nowhere. Exactly two planted values must survive the resize
    exactly; bilinear resampling would forge intermediates at the boundary."""

    # size=504 makes the transform NON-IDENTITY (scale 9.0, nonzero crop): at
    # the fixture's native 56 the gather is a 1:1 copy and a bilinear
    # resampler would be indistinguishable. The discontinuity sits at an odd
    # original column so an upscale blend has something to forge.
    scene = fixture_scene(cameras=(0, 1), times=(0,), size=504)
    with torch.no_grad():
        planted = torch.full_like(scene.depth, 10.0)
        planted[..., :27] = 1.0
        scene.depth.copy_(planted)
        scene.depth0.copy_(scene.depth[:, 0])

    dumped_kubric._attach_view_geometry(
        scene, input_depth_max=24.0, input_camera_vectors=False
    )

    allowed = (np.array([1.0, 10.0], dtype=np.float64) / 24.0).astype(np.float32)
    for view in scene.views:
        channel = view["depth_map"][0, 0].numpy()
        validity = view["depth_map"][0, 1].numpy()
        assert (validity == 1.0).all()
        assert np.isin(channel, allowed).all()


def test_depth_input_sanitises_invalidates_and_scales():
    """Landmine 8 plus the settled far-depth semantics: --kubric_max_depth is
    an INVALIDITY threshold. nan, inf, negative, zero and beyond-max all read
    (0, 0); at-max saturates valid at 1.0; metric scale is kept, so two maxes
    scale one shared depth in exact ratio and nothing is median-normalised."""

    scene = fixture_scene(cameras=(0, 1), times=(0,), size=56)
    observation = scene.observations[0]
    rows, columns = observation.image_transform.output_to_original_indices()
    specials = [float("nan"), float("inf"), -3.0, 0.0, 30.0, 24.0]
    with torch.no_grad():
        scene.depth.fill_(12.0)
        # Well-separated output rows, one shared column, so the planted
        # original pixels cannot collide under the rounding gather.
        for index, value in enumerate(specials):
            scene.depth[0, 0, 0, rows[index * 8], columns[0]] = value
        scene.depth0.copy_(scene.depth[:, 0])

    dumped_kubric._attach_view_geometry(
        scene, input_depth_max=24.0, input_camera_vectors=False
    )
    channel = scene.views[0]["depth_map"][0, 0]
    validity = scene.views[0]["depth_map"][0, 1]
    for index in range(5):
        assert float(channel[index * 8, 0]) == 0.0
        assert float(validity[index * 8, 0]) == 0.0
    assert float(channel[5 * 8, 0]) == 1.0
    assert float(validity[5 * 8, 0]) == 1.0
    assert float(channel[20, 30]) == 0.5
    assert float(validity[20, 30]) == 1.0

    other = fixture_scene(cameras=(0, 1), times=(0,), size=56)
    with torch.no_grad():
        other.depth.fill_(12.0)
        other.depth0.fill_(12.0)
    dumped_kubric._attach_view_geometry(
        other, input_depth_max=12.0, input_camera_vectors=False
    )
    assert float(other.views[0]["depth_map"][0, 0, 20, 30]) == 1.0


def test_camera_vector_is_model_grid_c2w_with_principal_point():
    """Landmines 11 and 12, index by index against independent recomputation.

    vec[:3] is the camera CENTRE -- true only for the camera-to-world pose, so
    a forgotten inversion is finite, plausible and fails here. vec[3:7] is the
    xyzw quaternion (real part last, mat_to_quat's contract) of the c2w
    rotation. vec[7:9] are the (h, w)-ORDERED fovs of the model-grid focals;
    vec[9:11] switch convention to the (x, y)-ordered principal point -- the
    deliberate mid-vector switch the loader comments. The projection property
    then pins the whole K_model against the transform, and the 9-dim round
    trip shows why terms 9-10 exist at all.
    """

    # size=504: a NON-IDENTITY transform (scale 9.0, nonzero crop), or the
    # model-grid half of the encoding is vacuous -- raw original intrinsics,
    # swapped (h, w) and a wrong-axis principal point all pass at identity.
    scene = fixture_scene(
        rotated_camera=1,
        cameras=(0, 1),
        times=(0, 1, 2, 3),
        size=504,
        input_camera_vectors=True,
    )

    for observation, view in zip(scene.observations, scene.views):
        vec = view["camera_vector"][0].double()
        transform = observation.image_transform
        intrinsics = scene.intrinsics[
            observation.camera, observation.original_time
        ].double()
        w2c = scene.extrinsics_world_to_camera[
            observation.camera, observation.original_time
        ].double()
        rotation = w2c[:3, :3]
        translation = w2c[:3, 3]
        centre = -rotation.T @ translation

        np.testing.assert_allclose(vec[:3].numpy(), centre.numpy(), atol=1e-5)

        expected_quat = _quaternion_xyzw(rotation.T.numpy())
        candidate = vec[3:7].numpy()
        assert np.allclose(candidate, expected_quat, atol=1e-5) or np.allclose(
            candidate, -expected_quat, atol=1e-5
        )

        principal = transform.original_to_output(
            np.array([[float(intrinsics[0, 2]), float(intrinsics[1, 2])]])
        )[0]
        fx_out = float(intrinsics[0, 0]) * transform.scale_x
        fy_out = float(intrinsics[1, 1]) * transform.scale_y
        height_out = transform.output_height
        width_out = transform.output_width
        assert float(vec[7]) == pytest.approx(
            2 * math.atan((height_out / 2) / fy_out), abs=1e-5
        )
        assert float(vec[8]) == pytest.approx(
            2 * math.atan((width_out / 2) / fx_out), abs=1e-5
        )
        assert float(vec[9]) == pytest.approx(principal[0] / width_out, abs=1e-5)
        assert float(vec[10]) == pytest.approx(principal[1] / height_out, abs=1e-5)

        # Projection consistency: the model-grid projection of a world point
        # equals the transform of its original-grid projection -- the same
        # coordinates the track head is supervised in. Forgetting the crop in
        # the principal point, crop-only or resize-only focals, or swapped
        # h/w each break this.
        world_point = centre + rotation.T @ torch.tensor(
            [0.1, -0.2, 2.0], dtype=torch.float64
        )
        camera_point = rotation @ world_point + translation
        model_k = torch.tensor(
            [
                [fx_out, 0.0, principal[0]],
                [0.0, fy_out, principal[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        original_projection = intrinsics @ camera_point
        original_xy = (original_projection[:2] / original_projection[2]).numpy()
        model_projection = model_k @ camera_point
        model_xy = (model_projection[:2] / model_projection[2]).numpy()
        expected_xy = transform.original_to_output(original_xy[None])[0]
        np.testing.assert_allclose(model_xy, expected_xy, atol=1e-9)

        # The 9-dim round trip recovers pose and focals but CENTRES the
        # principal point -- exactly the information terms 9-10 carry.
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            vec[None, None, :9].float(), (height_out, width_out)
        )
        np.testing.assert_allclose(
            extrinsic[0, 0, :3, 3].numpy(), centre.numpy(), atol=1e-4
        )
        assert float(intrinsic[0, 0, 0, 2]) == pytest.approx(width_out / 2, abs=1e-3)
        assert float(intrinsic[0, 0, 1, 2]) == pytest.approx(height_out / 2, abs=1e-3)


def test_model_grid_intrinsics_match_the_direct_affine_formula():
    """The settled K_model pin: production derives every model-grid intrinsic
    entry through original_to_output, the repo's ONE spelling of the
    scale/crop affine; the readable direct formula lives HERE, as the
    expectation, on a transform with non-trivial scale and non-zero crop in
    both axes. If ImageTransform ever stops being affine, this is the test
    that fails."""

    transform = dumped_kubric.ImageTransform(
        original_height=96,
        original_width=128,
        resized_height=64,
        resized_width=86,
        crop_top=5,
        crop_left=9,
        output_height=54,
        output_width=68,
    )
    fx, fy, cx, cy = 100.0, 90.0, 63.5, 47.5

    principal = transform.original_to_output(np.array([[cx, cy]]))[0]
    fx_out = transform.original_to_output(np.array([[cx + fx, cy]]))[0][0] - principal[0]
    fy_out = transform.original_to_output(np.array([[cx, cy + fy]]))[0][1] - principal[1]

    np.testing.assert_allclose(fx_out, fx * transform.scale_x, rtol=1e-12)
    np.testing.assert_allclose(fy_out, fy * transform.scale_y, rtol=1e-12)
    np.testing.assert_allclose(
        principal[0], cx * transform.scale_x - transform.crop_left, rtol=1e-12
    )
    np.testing.assert_allclose(
        principal[1], cy * transform.scale_y - transform.crop_top, rtol=1e-12
    )


def test_single_flag_arms_and_the_gradient_guard():
    """Landmine 4 under fork (b), both directions.

    A frozen-and-unfed module satisfies both window guards by construction --
    which is what licenses always-construct. Trainable-but-unfed is LOUD: the
    first window close raises. The remaining silent case, fed-but-frozen, is
    the parameter-set count test's job (test_runtime), not a gradient
    guard's: frozen parameters receive no gradient at all.
    """

    torch.manual_seed(0)
    encoder = _small_encoder()
    images = torch.randn(1, 4, 3, 14, 14)
    depth = torch.rand(1, 4, 2, 14, 14) + 0.5
    vectors = torch.randn(1, 4, 11)

    encoder.requires_grad_(False)
    encoder.depth_patch_embed.requires_grad_(True)
    out = _run_pass_through(encoder, images, depth_maps=depth)
    out.sum().backward()
    runtime.assert_trainable_gradients_finite(encoder)
    runtime.assert_frozen_gradients_absent(encoder)

    # Camera-only, fed AND trainable: the injected token must actually reach
    # the taps, or a camera-only run trains an inert projection under a
    # healthy-looking loss curve. The nonzero gradient is the pin.
    encoder.zero_grad(set_to_none=True)
    encoder.requires_grad_(False)
    encoder.camera_proj.requires_grad_(True)
    out = _run_pass_through(encoder, images, camera_vectors=vectors)
    out.sum().backward()
    runtime.assert_trainable_gradients_finite(encoder)
    runtime.assert_frozen_gradients_absent(encoder)
    assert float(encoder.camera_proj.weight.grad.abs().sum()) > 0

    encoder.zero_grad(set_to_none=True)
    encoder.depth_patch_embed.requires_grad_(True)
    out = _run_pass_through(encoder, images, depth_maps=depth)
    out.sum().backward()
    with pytest.raises(RuntimeError, match="missing gradients"):
        runtime.assert_trainable_gradients_finite(encoder)


def test_move_views_to_cuda_moves_geometry_keys_when_present():
    """The trio stays mandatory; the geometry pair is optional -- present it
    moves, absent it is not an error."""

    class _Recorder:
        def __init__(self):
            self.moves = 0

        def to(self, device, non_blocking=False):
            assert device == "cuda" and non_blocking
            self.moves += 1
            return self

    full = {
        key: _Recorder()
        for key in ("img", "time_index", "track_query_idx", "depth_map", "camera_vector")
    }
    bare = {key: _Recorder() for key in ("img", "time_index", "track_query_idx")}

    runtime.move_views_to_cuda([full, bare])

    assert sum(value.moves for value in full.values()) == 5
    assert sum(value.moves for value in bare.values()) == 3


def test_geometry_signatures_are_keyword_only():
    """Flag-off bit-identity's signature half, plus the config.json pin:
    Arc.__init__ takes no geometry parameter, so the Hub mixin can never
    serialise one (landmine 15)."""

    for function, names in (
        (Arc.encode_features, ("depth_maps", "camera_vectors")),
        (Arc._forward, ("depth_maps", "camera_vectors")),
        (DinoVisionTransformer.prepare_tokens_with_masks, ("depth_maps",)),
    ):
        for name in names:
            parameter = inspect.signature(function).parameters[name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default is None

    for function in (Arc.set_freeze, runtime.assert_trainable_parameter_set):
        for name in ("depth_input", "camera_input", "refine"):
            parameter = inspect.signature(function).parameters[name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default is False

    build_scene_parameters = inspect.signature(dumped_kubric.build_scene).parameters
    assert (
        build_scene_parameters["input_depth_max"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert build_scene_parameters["input_depth_max"].default is None
    assert build_scene_parameters["input_camera_vectors"].default is False

    init_parameters = inspect.signature(Arc.__init__).parameters
    for name in ("depth_input", "camera_input", "depth_maps", "camera_vectors"):
        assert name not in init_parameters


def test_forward_threads_geometry_views_to_the_backbone():
    """The Arc-level route pin. Recording is weight-independent, so an
    argument dropped at forward, _forward or encode_features fails here even
    under zero init -- the hole no output comparison can see. The img stays
    3-channel beside the depth key, which is landmine 10's witness."""

    model = _arc_shell(max_time_indices=32)
    model.backbone = _FakeBackbone()
    model.head = _FakeReconstructionHead()
    model.cam_dec = _FakeCameraDecoder()
    model.motion_decoder = _FakeMotionDecoder()
    model.track_head = _FakeTrackHead()

    views = [
        {
            "img": torch.zeros(1, 3, 2, 2),
            "time_index": torch.tensor([view]),
            "track_query_idx": torch.tensor([0]),
            "depth_map": torch.full((1, 2, 2, 2), float(view + 1)),
            "camera_vector": torch.full((1, 11), float(view + 1)),
        }
        for view in range(4)
    ]
    model(views, force_no_output_conversion=True)

    expected_depth = torch.stack([view["depth_map"] for view in views], dim=1)
    expected_camera = torch.stack([view["camera_vector"] for view in views], dim=1)
    assert torch.equal(model.backbone.seen_depth_maps, expected_depth)
    assert torch.equal(model.backbone.seen_camera_vectors, expected_camera)

    control = [
        {"img": torch.zeros(1, 3, 2, 2), "track_query_idx": torch.tensor([0])}
        for _ in range(4)
    ]
    model(control, force_no_output_conversion=True)
    assert model.backbone.seen_depth_maps is None
    assert model.backbone.seen_camera_vectors is None
