import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["ATTN_BACKEND"] = "xformers"
os.environ.setdefault("CONDA_PREFIX", sys.prefix)

PROJECT_ROOT = Path(__file__).resolve().parent
SAM3D_OBJECTS_ROOT = PROJECT_ROOT.parent / "sam-3d-objects"
LOCAL_PACKAGE_PATHS = [
    PROJECT_ROOT / "oneposeviagen" / "SpaTrackerV2",
    PROJECT_ROOT / "oneposeviagen" / "Amodal3R",
    PROJECT_ROOT / "oneposeviagen" / "trellis",
    PROJECT_ROOT / "oneposeviagen" / "fpose",
]
for package_path in reversed(LOCAL_PACKAGE_PATHS):
    if package_path.exists():
        sys.path.insert(0, str(package_path))

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from app_3rd.spatrack_utils.infer_track import get_points_on_a_grid, get_tracker_predictor
from fpose.recover_scale import recover_scale
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from oneposeviagen.scripts.estimate_poses import estimate_poses
from oneposeviagen.scripts.render_normals import render_high_model_to_normal_video
from torchvision.utils import save_image


MAX_FRAMES_OFFLINE = 50
MAX_SEED = np.iinfo(np.int32).max
VIDEO_FPS = 10



def sorted_image_paths(directory):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in Path(directory).iterdir() if p.suffix.lower() in suffixes]

    def sort_key(path):
        try:
            return 0, int(path.stem)
        except ValueError:
            return 1, path.name

    return [str(path) for path in sorted(paths, key=sort_key)]


def create_workspace(output_dir=None):
    if output_dir is None:
        output_dir = Path("temp_local") / f"dir_infer_{uuid.uuid4().hex[:8]}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def preprocess_mask_tensor(mask_tensor, target_size=518):
    _, height, width = mask_tensor.shape
    new_width = target_size
    new_height = round(height * (new_width / width) / 14) * 14
    mask_tensor = torch.nn.functional.interpolate(
        mask_tensor.unsqueeze(0),
        size=(new_height, new_width),
        mode="nearest",
    ).squeeze(0)
    if new_height > target_size:
        start_y = (new_height - target_size) // 2
        mask_tensor = mask_tensor[:, start_y:start_y + target_size, :]
    return mask_tensor


def prepare_workspace_from_dirs(rgb_dir, mask_dir, workspace, frame_stride=1, max_frames=MAX_FRAMES_OFFLINE):
    rgb_paths = sorted_image_paths(rgb_dir)
    mask_paths = sorted_image_paths(mask_dir)
    if not rgb_paths:
        raise ValueError(f"No images found in rgb_dir: {rgb_dir}")
    if not mask_paths:
        raise ValueError(f"No masks found in mask_dir: {mask_dir}")
    if len(rgb_paths) != len(mask_paths):
        raise ValueError(f"Frame/mask count mismatch: {len(rgb_paths)} frames vs {len(mask_paths)} masks")

    rgb_paths = rgb_paths[::frame_stride][:max_frames]
    mask_paths = mask_paths[::frame_stride][:max_frames]

    workspace = Path(workspace)
    out_rgb_dir = workspace / "rgb"
    out_mask_dir = workspace / "masks"
    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    image_tensors = []
    mask_tensors = []
    for rgb_path, mask_path in zip(rgb_paths, mask_paths):
        image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        height, width = image.shape[:2]
        scale = 336 / min(height, width)
        if scale < 1:
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        image_tensors.append(torch.from_numpy(image).permute(2, 0, 1).float())
        mask_tensors.append(torch.from_numpy((mask > 127).astype(np.float32))[None])

    image_tensor = preprocess_image(torch.stack(image_tensors))
    mask_tensor = torch.stack([preprocess_mask_tensor(mask) for mask in mask_tensors])

    for frame_id, (image, mask) in enumerate(zip(image_tensor, mask_tensor)):
        save_image((image / 255).clamp(0, 1), out_rgb_dir / f"{frame_id:06d}.jpg")
        mask_image = (mask.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8) * 255
        cv2.imwrite(str(out_mask_dir / f"{frame_id:06d}.png"), mask_image)

    return str(out_rgb_dir), str(out_mask_dir)


class DirInferenceModels:
    def __init__(self):
        print("🚀 Loading SpatialTrackerV2 models...")
        self.vggt4track_model = VGGT4Track.from_pretrained(
            "checkpoints/OnePoseViaGen/SpatialTrackerV2/vggt_front"
        ).eval().to("cuda")
        self.tracker_model_offline = Predictor.from_pretrained(
            "checkpoints/OnePoseViaGen/SpatialTrackerV2/tracker_offline"
        ).eval()
        self.tracker_model_online = Predictor.from_pretrained(
            "checkpoints/OnePoseViaGen/SpatialTrackerV2/tracker_online"
        ).eval()

        self.amodal3r_pipeline = None
        self.hi3dgen_pipeline = None
        print("✅ Models loaded.")


def load_prepared_rgb_tensor(rgb_dir):
    image_paths = sorted_image_paths(rgb_dir)
    tensors = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read prepared image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensors.append(torch.from_numpy(image).permute(2, 0, 1).float())
    return torch.stack(tensors).cuda()


def convert_video_to_mp4(input_path, output_path):
    command = [
        "/usr/bin/ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-y",
        output_path,
    ]
    subprocess.run(command, check=True)


def convert_depth_images_to_video(file_paths, output_video_path, fps=VIDEO_FPS):
    if not file_paths:
        raise ValueError("file_paths is empty")
    first_image = cv2.imread(file_paths[0], cv2.IMREAD_UNCHANGED)
    if first_image is None:
        raise ValueError(f"Failed to load first depth image: {file_paths[0]}")
    height, width = first_image.shape
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), isColor=True)
    all_depths = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in file_paths]
    all_depths = [depth for depth in all_depths if depth is not None]
    for depth_image in all_depths:
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        writer.write(cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR))
    writer.release()


def run_tracker_from_workspace(models, workspace, grid_size=50, vo_points=756, mode="offline"):
    rgb_dir = os.path.join(workspace, "rgb")
    mask_dir = os.path.join(workspace, "masks")
    out_dir = os.path.join(workspace, "results")
    os.makedirs(out_dir, exist_ok=True)

    tracker_model = models.tracker_model_offline if mode == "offline" else models.tracker_model_online
    tracker_model, tracker_viser = get_tracker_predictor(
        out_dir,
        vo_points=vo_points,
        tracker_model=tracker_model.cuda(),
    )

    video_tensor = load_prepared_rgb_tensor(rgb_dir)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        predictions = models.vggt4track_model(video_tensor[None].cuda() / 255)
        extrinsic = predictions["poses_pred"]
        intrinsic = predictions["intrs"]
        depth_map = predictions["points_map"][..., 2]
        depth_conf = predictions["unc_metric"]

    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

    mask_path = sorted_image_paths(mask_dir)[0]
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (video_tensor.shape[3], video_tensor.shape[2]), interpolation=cv2.INTER_NEAREST) > 127

    frame_height, frame_width = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_height, frame_width), device="cuda")
    grid_pts_int = grid_pts[0].long()
    mask_values = mask[grid_pts_int.cpu()[..., 1], grid_pts_int.cpu()[..., 0]]
    grid_pts = grid_pts[:, mask_values]
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].cpu().numpy()
    print(f"Query points shape: {query_xyt.shape}")

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video,
        ) = tracker_model.forward(
            video_tensor,
            depth=depth_tensor,
            intrs=intrs,
            extrs=extrs,
            queries=query_xyt,
            fps=1,
            full_point=False,
            iters_track=4,
            query_no_BA=True,
            fixed_cam=False,
            stage=1,
            unc_metric=unc_metric,
            support_frame=len(video_tensor) - 1,
            replace_ratio=0.2,
        )

        tracker_viser.visualize(
            video=video[None],
            tracks=track2d_pred[None][..., :2],
            visibility=vis_pred[None],
            filename="test",
        )

        result = {
            "coords": (
                torch.einsum("tij,tnj->tni", c2w_traj[:, :3, :3].cpu(), track3d_pred[:, :, :3].cpu())
                + c2w_traj[:, :3, 3][:, None, :].cpu()
            ).numpy(),
            "extrinsics": torch.inverse(c2w_traj).cpu().numpy(),
            "intrinsics": intrs.cpu().numpy(),
            "depths": point_map[:, 2, ...].cpu().numpy(),
            "video": video_tensor.cpu().numpy() / 255,
            "visibs": vis_pred.cpu().numpy(),
            "confs": conf_pred.cpu().numpy(),
            "confs_depth": conf_depth.cpu().numpy(),
        }

    depth_dir = os.path.join(workspace, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    depth_names = []
    for frame_id, depth_map_save in enumerate(result["depths"]):
        depth_path = os.path.join(depth_dir, f"{frame_id:06d}.png")
        cv2.imwrite(depth_path, (depth_map_save * 1000).astype("uint16"))
        depth_names.append(depth_path)

    with open(os.path.join(workspace, "intrinsics.json"), "w") as f:
        json.dump({str(i): result["intrinsics"][i].tolist() for i in range(len(result["intrinsics"]))}, f, indent=2)
    np.savez(os.path.join(out_dir, "result.npz"), **result)

    depth_video_path = os.path.join(depth_dir, "depth.mp4")
    depth_video_path_new = os.path.join(depth_dir, "depth_new.mp4")
    convert_depth_images_to_video(depth_names, depth_video_path)
    convert_video_to_mp4(depth_video_path, depth_video_path_new)
    os.remove(depth_video_path)
    return depth_video_path_new


def generate_final_mask(seg_path, depth_path, area_threshold=100, max_iter=50):
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    object_mask = seg > 128
    object_depths = depth[object_mask]
    min_depth = np.percentile(object_depths, 90)
    occluder_mask = (depth < min_depth) & (~object_mask)
    kernel = np.ones((3, 3), np.uint8)
    occluder_mask = cv2.erode(occluder_mask.astype(np.uint8), kernel, iterations=1)
    x, y, width, height = cv2.boundingRect(object_mask.astype(np.uint8))
    occluder_mask_bbox = np.zeros_like(occluder_mask)
    occluder_mask_bbox[y:y + height, x:x + width] = occluder_mask[y:y + height, x:x + width]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(occluder_mask_bbox, connectivity=8)
    filtered_mask = np.zeros_like(occluder_mask_bbox)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= area_threshold:
            filtered_mask[labels == label_id] = 1
    dilated_occluder = filtered_mask.copy()
    for _ in range(max_iter):
        if np.any((dilated_occluder > 0) & object_mask):
            break
        dilated_occluder = cv2.dilate(dilated_occluder, kernel, iterations=1)
    gap_mask = (dilated_occluder > 0) & (~filtered_mask.astype(bool)) & (~object_mask)
    filtered_mask[gap_mask] = 1
    final_mask = np.ones_like(seg, dtype=np.uint8) * 255
    final_mask[object_mask] = 188
    final_mask[filtered_mask > 0] = 0
    return Image.fromarray(final_mask)


def mask_image(rgb_path, mask_path):
    input_image = Image.open(rgb_path).convert("RGB")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    rgb_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    result_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_binary)
    output = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    bbox = np.argwhere(mask_binary > 0.8 * 255)
    if len(bbox) == 0:
        return output
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.2)
    crop_box = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(crop_box).resize((518, 518), Image.LANCZOS)
    return output


def mask_image_and_mask(rgb_path, mask_path):
    rgb_image = mask_image(rgb_path, mask_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    bbox = np.argwhere(mask_binary > 0.8 * 255)
    if len(bbox) == 0:
        return rgb_image, Image.fromarray(mask_binary)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.2)
    crop_box = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    cropped_mask = Image.fromarray(mask_binary).crop(crop_box).resize((518, 518), Image.NEAREST)
    return rgb_image, cropped_mask


def run_sam3d_subprocess(image_path, mask_path, mesh_path, high_mesh_path, video_path, splat_path, seed):
    helper_script = PROJECT_ROOT / "scripts" / "run_sam3d_stage.py"
    sam3d_python = os.environ.get("SAM3D_PYTHON")
    if sam3d_python:
        command = [sam3d_python, str(helper_script)]
    else:
        sam3d_env = os.environ.get("SAM3D_CONDA_ENV", "sam3d-objects")
        command = ["conda", "run", "-n", sam3d_env, "python", str(helper_script)]

    command.extend([
        "--sam3d-root", str(SAM3D_OBJECTS_ROOT),
        "--image", image_path,
        "--mask", mask_path,
        "--mesh-path", mesh_path,
        "--high-mesh-path", high_mesh_path,
        "--video-path", video_path,
        "--splat-path", splat_path,
        "--seed", str(seed),
    ])
    subprocess.run(command, check=True)


def generate_3d_with_sam3d(models, image, mask, workspace, export_format="obj", seed=-1):
    if seed == -1:
        seed = np.random.randint(0, MAX_SEED)

    model_dir = os.path.join(workspace, "model")
    model_middle_dir = os.path.join(model_dir, "middle_file")
    os.makedirs(model_middle_dir, exist_ok=True)
    output_id = str(uuid.uuid4())
    video_path = os.path.join(model_middle_dir, f"{output_id}_preview.mp4")
    mesh_path = os.path.join(model_dir, f"model.{export_format}")
    high_mesh_path = os.path.join(model_middle_dir, f"{output_id}_high_mesh.obj")
    splat_path = os.path.join(model_middle_dir, f"{output_id}_splat.ply")
    sam3d_image_path = os.path.join(model_middle_dir, f"{output_id}_sam3d_input.png")
    sam3d_mask_path = os.path.join(model_middle_dir, f"{output_id}_sam3d_mask.png")

    image_pil = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.array(image).astype(np.uint8)).convert("RGB")
    mask_np = np.array(mask if not isinstance(mask, Image.Image) else mask.convert("L"))
    mask_np = mask_np == 188 if np.any(mask_np == 188) else mask_np > 0
    mask_pil = Image.fromarray(mask_np.astype(np.uint8) * 255)

    image_pil.save(sam3d_image_path)
    mask_pil.save(sam3d_mask_path)
    run_sam3d_subprocess(sam3d_image_path, sam3d_mask_path, mesh_path, high_mesh_path, video_path, splat_path, seed)
    return video_path, mesh_path, high_mesh_path


def generate_3d(models, image, mask, workspace, export_format="obj", seed=-1,
                ss_guidance_strength=7.5, ss_sampling_steps=12,
                slat_guidance_strength=3, slat_sampling_steps=12,
                is_occluded=False):
    return generate_3d_with_sam3d(models, image, mask, workspace, export_format=export_format, seed=seed)


def recover_true_scale(normal_model_path, anchor_depth_name, anchor_intrinsic, anchor_image_name, anchor_mask_name, output_dir):
    anchor_dir = os.path.join(output_dir, "anchor_file")
    os.makedirs(anchor_dir, exist_ok=True)
    intrinsic_file = os.path.join(anchor_dir, "intrinsic.txt")
    np.savetxt(intrinsic_file, anchor_intrinsic, fmt="%.6f")
    mid_dir = os.path.join(output_dir, "mid_files")
    os.makedirs(mid_dir, exist_ok=True)
    scaled_mesh_path = os.path.join(mid_dir, "scaled_mesh.obj")
    scaled_mesh, pose, final_scale = recover_scale(
        normal_model_path,
        anchor_depth_name,
        anchor_image_name,
        anchor_mask_name,
        intrinsic_file,
        "test",
        mid_dir,
    )
    scaled_mesh.export(scaled_mesh_path)
    return scaled_mesh_path, final_scale, pose


def choose_anchor_index(mask_names):
    if not mask_names:
        raise ValueError("No masks available for automatic anchor selection")

    best_index = 0
    best_area = -1
    for index, mask_path in enumerate(mask_names):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask for anchor selection: {mask_path}")
        area = int(np.count_nonzero(mask > 127))
        if area > best_area:
            best_index = index
            best_area = area

    print(f"🎯 Auto-selected anchor_index={best_index} with mask area={best_area}")
    return best_index


def generate_model_from_workspace(workspace, anchor_index=None, seed=0, randomize_seed=False,
                                  ss_guidance_strength=7.5, ss_sampling_steps=12,
                                  slat_guidance_strength=3, slat_sampling_steps=12,
                                  is_occluded=False):
    rgb_names = sorted_image_paths(os.path.join(workspace, "rgb"))
    mask_names = sorted_image_paths(os.path.join(workspace, "masks"))
    if anchor_index is None:
        anchor_index = choose_anchor_index(mask_names)
    if not 0 <= anchor_index < len(rgb_names):
        raise ValueError(f"anchor_index {anchor_index} is out of range for {len(rgb_names)} frames")

    model_dir = os.path.join(workspace, "model")
    os.makedirs(model_dir, exist_ok=True)
    rgb_image, final_mask = mask_image_and_mask(rgb_names[anchor_index], mask_names[anchor_index])
    final_mask.save(os.path.join(model_dir, "final_mask.png"))

    actual_seed = np.random.randint(0, MAX_SEED) if randomize_seed else seed
    return generate_3d(
        None,
        rgb_image,
        final_mask,
        workspace,
        "ply",
        actual_seed,
        ss_guidance_strength,
        ss_sampling_steps,
        slat_guidance_strength,
        slat_sampling_steps,
        is_occluded,
    )


def rescale_model_from_workspace(workspace, anchor_index, mesh_path, high_mesh_path):
    rgb_names = sorted_image_paths(os.path.join(workspace, "rgb"))
    mask_names = sorted_image_paths(os.path.join(workspace, "masks"))
    depth_names = sorted_image_paths(os.path.join(workspace, "depth"))
    if not 0 <= anchor_index < len(rgb_names):
        raise ValueError(f"anchor_index {anchor_index} is out of range for {len(rgb_names)} frames")
    with open(os.path.join(workspace, "intrinsics.json"), "r") as f:
        intrinsic = json.load(f)[str(anchor_index)]

    model_dir = os.path.join(workspace, "model")
    scaled_model_path, scale, _ = recover_true_scale(
        mesh_path,
        depth_names[anchor_index],
        intrinsic,
        rgb_names[anchor_index],
        mask_names[anchor_index],
        model_dir,
    )
    high_mesh = trimesh.load(high_mesh_path)
    high_mesh.vertices = high_mesh.vertices * scale
    high_mesh.export(high_mesh_path)
    return scaled_model_path, high_mesh_path


def estimate_query_poses_from_workspace(workspace, scaled_model_path, high_mesh_path):
    pose_debug_dir = os.path.join(workspace, "pose_debug")
    pose_dir = os.path.join(workspace, "pose_result")
    os.makedirs(pose_debug_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    rgb_names = sorted_image_paths(os.path.join(workspace, "rgb"))
    depth_names = sorted_image_paths(os.path.join(workspace, "depth"))
    mask_names = sorted_image_paths(os.path.join(workspace, "masks"))
    with open(os.path.join(workspace, "intrinsics.json"), "r") as f:
        intrinsics_dict = json.load(f)
    intrinsics = [intrinsics_dict[str(frame_id)] for frame_id in range(len(rgb_names))]

    npz_path = os.path.join(workspace, "results", "result.npz")
    poses = estimate_poses(
        npz_path,
        rgb_names,
        depth_names,
        mask_names,
        intrinsics,
        scaled_model_path,
        pose_debug_dir,
        debug=0,
        est_refine_iter=5,
    )

    poses_file_path = os.path.join(pose_dir, "poses.json")
    with open(poses_file_path, "w") as f:
        json.dump({str(frame_id): poses[frame_id].tolist() for frame_id in range(len(poses))}, f, indent=2)

    normal_video_path = os.path.join(pose_dir, "normal_video.mp4")
    normal_video_path_new = os.path.join(pose_dir, "normal_video_new.mp4")
    render_high_model_to_normal_video(poses, rgb_names, intrinsics, high_mesh_path, normal_video_path, fps=VIDEO_FPS, device="cuda")
    convert_video_to_mp4(normal_video_path, normal_video_path_new)
    os.remove(normal_video_path)
    return normal_video_path_new, poses_file_path


def run_inference_from_dirs(rgb_dir, mask_dir, output_dir=None, frame_stride=1, max_frames=MAX_FRAMES_OFFLINE,
                            grid_size=50, vo_points=756, mode="offline", anchor_index=None, seed=0,
                            randomize_seed=False, ss_guidance_strength=7.5, ss_sampling_steps=12,
                            slat_guidance_strength=3, slat_sampling_steps=12, is_occluded=False):
    workspace = create_workspace(output_dir)
    prepare_workspace_from_dirs(rgb_dir, mask_dir, workspace, frame_stride=frame_stride, max_frames=max_frames)
    if anchor_index is None:
        anchor_index = choose_anchor_index(sorted_image_paths(os.path.join(workspace, "masks")))
    model_video, mesh_path, high_mesh_path = generate_model_from_workspace(
        workspace,
        anchor_index=anchor_index,
        seed=seed,
        randomize_seed=randomize_seed,
        ss_guidance_strength=ss_guidance_strength,
        ss_sampling_steps=ss_sampling_steps,
        slat_guidance_strength=slat_guidance_strength,
        slat_sampling_steps=slat_sampling_steps,
        is_occluded=is_occluded,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    models = DirInferenceModels()
    depth_video = run_tracker_from_workspace(models, workspace, grid_size=grid_size, vo_points=vo_points, mode=mode)
    scaled_model_path, high_mesh_path = rescale_model_from_workspace(workspace, anchor_index, mesh_path, high_mesh_path)
    pose_video, poses_json = estimate_query_poses_from_workspace(workspace, scaled_model_path, high_mesh_path)
    outputs = {
        "workspace": workspace,
        "depth_video": depth_video,
        "model_video": model_video,
        "scaled_model_path": scaled_model_path,
        "high_mesh_path": high_mesh_path,
        "pose_video": pose_video,
        "poses_json": poses_json,
        "tracking_npz": os.path.join(workspace, "results", "result.npz"),
        "anchor_index": anchor_index,
    }
    with open(os.path.join(workspace, "outputs.json"), "w") as f:
        json.dump(outputs, f, indent=2)
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Run OnePoseviaGen from pre-extracted RGB frames and masks.")
    parser.add_argument("--rgb-dir", required=True, help="Folder containing RGB frames.")
    parser.add_argument("--mask-dir", required=True, help="Folder containing binary masks aligned with RGB frames.")
    parser.add_argument("--output-dir", default=None, help="Workspace/output directory. Defaults to temp_local/dir_infer_*.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every N-th frame.")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES_OFFLINE, help="Maximum number of frames to use.")
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--vo-points", type=int, default=756)
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--anchor-index", type=int, default=None, help="Prepared frame index used for 3D generation and scale recovery. Defaults to the mask with the largest foreground area.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-seed", action="store_true")
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-sampling-steps", type=int, default=12)
    parser.add_argument("--slat-guidance-strength", type=float, default=3)
    parser.add_argument("--slat-sampling-steps", type=int, default=12)
    parser.add_argument("--is-occluded", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    result = run_inference_from_dirs(**vars(args))
    print(json.dumps(result, indent=2))
    print(f"Done in {time.time() - start_time:.1f}s")
