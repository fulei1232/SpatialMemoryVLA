import numpy as np

from evaluation.libero.hard_cases import (
    RelocationConfig,
    TemporalOcclusionConfig,
    apply_temporal_occlusion,
    get_object_pose,
    relocate_object,
    set_counterfactual_relation,
    unexpected_object_contacts,
)


def test_temporal_occlusion_has_visible_partial_full_and_recovery_phases():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    cfg = TemporalOcclusionConfig()
    visible, phase0, severity0 = apply_temporal_occlusion(image, 0, 100, cfg)
    partial, phase1, severity1 = apply_temporal_occlusion(image, 40, 100, cfg)
    full, phase2, severity2 = apply_temporal_occlusion(image, 60, 100, cfg)
    recovered, phase3, severity3 = apply_temporal_occlusion(image, 80, 100, cfg)
    assert (phase0, phase1, phase2, phase3) == (
        "visible_history", "partial_occlusion", "full_occlusion", "recovered_visible"
    )
    assert severity0 == severity3 == 0.0
    assert 0.0 < severity1 < 1.0 and severity2 == 1.0
    assert np.array_equal(visible, image) and np.array_equal(recovered, image)
    assert partial.sum() < image.sum() and full.sum() == 0


class FakeData:
    def __init__(self):
        self.poses = {
            "bowl_joint": np.asarray([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]),
            "plate_joint": np.asarray([0.2, 0.1, 0.8, 1.0, 0.0, 0.0, 0.0]),
        }

    def get_joint_qpos(self, name):
        return self.poses[name]

    def set_joint_qpos(self, name, pose):
        self.poses[name] = np.asarray(pose).copy()


class FakeSim:
    def __init__(self):
        self.data = FakeData()

    def forward(self):
        pass


class FakeObject:
    def __init__(self, joint):
        self.joints = [joint]


class FakeEnv:
    def __init__(self):
        self.sim = FakeSim()
        self.objects_dict = {"bowl": FakeObject("bowl_joint"), "plate": FakeObject("plate_joint")}

    def _post_process(self):
        pass

    def _update_observables(self, force=False):
        pass

    def _get_observations(self):
        return {"ok": True}


def test_relocation_changes_physical_pose():
    env = FakeEnv()
    _, old_pose, new_pose = relocate_object(env, RelocationConfig("bowl", 4, (0.10, -0.05, 0.0)))
    np.testing.assert_allclose(new_pose[:3] - old_pose[:3], [0.10, -0.05, 0.0])
    np.testing.assert_allclose(get_object_pose(env, "bowl"), new_pose)


def test_relocation_rejects_pose_outside_workspace_before_mutating_state():
    env = FakeEnv()
    old_pose = get_object_pose(env, "bowl")
    cfg = RelocationConfig(
        "bowl", 4, (0.10, 0.0, 0.0), workspace_x_bounds=(-0.05, 0.05)
    )
    try:
        relocate_object(env, cfg)
    except ValueError as exc:
        assert "outside workspace" in str(exc)
    else:
        raise AssertionError("Expected unsafe relocation to be rejected")
    np.testing.assert_allclose(get_object_pose(env, "bowl"), old_pose)


def test_support_surface_contacts_are_not_reported_as_collisions():
    contacts = [
        ("table_collision", "bowl_g1"),
        ("bowl_g2", "robot0_right_gripper"),
        ("bowl_g3", "plate_g1"),
    ]
    assert unexpected_object_contacts(contacts) == contacts[1:]


def test_counterfactual_pair_changes_only_relation_pose():
    env = FakeEnv()
    _, left = set_counterfactual_relation(env, "bowl", "plate", "left", 0.1)
    left_pose = get_object_pose(env, "bowl")
    _, right = set_counterfactual_relation(env, "bowl", "plate", "right", 0.1)
    right_pose = get_object_pose(env, "bowl")
    assert left["relation_label"] == "left" and right["relation_label"] == "right"
    assert left_pose[1] > get_object_pose(env, "plate")[1]
    assert right_pose[1] < get_object_pose(env, "plate")[1]
