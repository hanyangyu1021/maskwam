"""
    Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments,
    with additional binary object masks for the objects of interest.

    This script extends regenerate.py by using SegmentationRenderEnv to produce
    per-frame binary masks (1 = object of interest, 0 = everything else) for both
    the front camera (agentview) and the wrist camera (eye_in_hand).

    Usage:
        python platforms/libero/regenerate_mask_step2.py \
            --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
            --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
            --libero_target_dir <PATH TO TARGET DIR>

        Example (LIBERO-Spatial):
            python platforms/libero/regenerate_mask_step2.py \
                --libero_task_suite libero_spatial \
                --libero_raw_data_dir data/libero_spatial_v0/libero_spatial \
                --libero_target_dir data/libero_spatial_v1/libero_spatial_256_no_noops_mask
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import SegmentationRenderEnv


IMAGE_RESOLUTION = 256


def _patch_robosuite_segmentation_overflow():
    """Fix robosuite+MuJoCo3 uint8 overflow when decoding instance segmentation.

    robosuite's ``read_pixels`` does ``rgb[:,:,1] * 256`` on uint8 arrays, which
    overflows under NumPy 2 / MuJoCo 3.x and makes SegmentationRenderEnv unusable.
    """
    import mujoco
    import robosuite.utils.binding_utils as binding_utils

    if getattr(binding_utils, "_maskwam_seg_overflow_patched", False):
        return

    def _read_pixels(self, width, height, depth=False, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb_img = np.empty((height, width, 3), dtype=np.uint8)
        depth_img = np.empty((height, width), dtype=np.float32) if depth else None
        mujoco.mjr_readPixels(
            rgb=rgb_img, depth=depth_img, viewport=viewport, con=self.con
        )

        ret_img = rgb_img
        if segmentation:
            r = rgb_img[:, :, 0].astype(np.int32)
            g = rgb_img[:, :, 1].astype(np.int32)
            b = rgb_img[:, :, 2].astype(np.int32)
            seg_img = r + g * (2**8) + b * (2**16)
            seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
            seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)
            for i in range(self.scn.ngeom):
                geom = self.scn.geoms[i]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            ret_img = seg_ids[seg_img]

        if depth:
            return (ret_img, depth_img)
        return ret_img

    binding_utils.MjRenderContextOffscreen.read_pixels = _read_pixels
    binding_utils._maskwam_seg_overflow_patched = True


_patch_robosuite_segmentation_overflow()


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment with segmentation support."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_segmentations": "instance",
    }
    env = SegmentationRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def get_binary_obj_mask(env, segmentation_image):
    """
    Convert an instance segmentation image to a binary object mask.

    Uses env.get_segmentation_of_interest() which returns:
        1.0  for obj_of_interest pixels
       -1.0  for background (no object)
        0.0  for other objects

    We convert to pure binary: 1 = object of interest, 0 = everything else.
    """
    seg_of_interest = env.get_segmentation_of_interest(segmentation_image.copy())
    return (seg_of_interest > 0).astype(np.uint8)

def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset (with object masks)!")

    # Create target directory
    if os.path.isdir(args.libero_target_dir):
        user_input = input(f"Target directory already exists at path: {args.libero_target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: ")
        if user_input != 'y':
            exit()
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = f"{args.libero_target_dir}/{args.libero_task_suite}_metainfo.json"
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    task_ids = (
        [args.task_id]
        if getattr(args, "task_id", None) is not None
        else list(range(num_tasks_in_suite))
    )
    max_demos = getattr(args, "max_demos", None)

    for task_id in tqdm.tqdm(task_ids):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        demo_indices = list(range(len(orig_data.keys())))
        if max_demos is not None:
            demo_indices = demo_indices[:max_demos]
        for i in demo_indices:
            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            # Reset environment (this builds the segmentation ID mapping),
            # then set initial state and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []
            agentview_obj_masks = []
            eye_in_hand_obj_masks = []

            # Replay original demo actions in environment and record observations
            for _, action in enumerate(orig_actions):
                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    print(f"\tSkipping no-op action: {action}")
                    num_noops += 1
                    continue

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    # For all other timesteps, get state from environment and record it
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                # Record original action (from demo)
                actions.append(action)

                # Record data returned by environment
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )
                agentview_images.append(obs["agentview_image"][::-1,:])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"][::-1,:])

                # Extract binary object masks from segmentation (flip vertically to match RGB)
                agentview_seg = obs["agentview_segmentation_instance"].squeeze(-1)[::-1, :]
                eye_in_hand_seg = obs["robot0_eye_in_hand_segmentation_instance"].squeeze(-1)[::-1, :]
                agentview_obj_masks.append(get_binary_obj_mask(env, agentview_seg))
                eye_in_hand_obj_masks.append(get_binary_obj_mask(env, eye_in_hand_seg))

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                ep_data_grp = grp.create_group(f"demo_{i}")
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                # Binary object masks: 1 = object of interest, 0 = everything else
                obs_grp.create_dataset("agentview_obj_mask", data=np.stack(agentview_obj_masks, axis=0))
                obs_grp.create_dataset("eye_in_hand_obj_mask", data=np.stack(eye_in_hand_obj_masks, axis=0))
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)

                num_success += 1

            num_replays += 1

            # Record success/false and initial environment state in metainfo dict
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

            # Write metainfo dict to JSON file
            # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        # Close HDF5 files
        orig_data_file.close()
        new_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, default="libero_10", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial")
    parser.add_argument("--libero_raw_data_dir", type=str, required=True,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial")
    parser.add_argument("--libero_target_dir", type=str, required=True,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops_mask")
    parser.add_argument("--task_id", type=int, default=None,
                        help="If set, only regenerate this task id (for quick checks).")
    parser.add_argument("--max_demos", type=int, default=None,
                        help="If set, only regenerate the first N demos per task (for quick checks).")
    args = parser.parse_args()

    # Start data regeneration
    main(args)
