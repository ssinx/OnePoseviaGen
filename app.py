import os
os.environ['TORCH_CUDA_ARCH_LIST']='9.0'
os.environ['ATTN_BACKEND'] = 'xformers'
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGE_PATHS = [
    PROJECT_ROOT / "oneposeviagen" / "SpaTrackerV2",
    PROJECT_ROOT / "oneposeviagen" / "SAM2-in-video",
    PROJECT_ROOT / "oneposeviagen" / "Amodal3R",
    PROJECT_ROOT / "oneposeviagen" / "trellis",
    PROJECT_ROOT / "oneposeviagen" / "fpose",
]
for package_path in reversed(LOCAL_PACKAGE_PATHS):
    if package_path.exists():
        sys.path.insert(0, str(package_path))

import gradio as gr
import json
import numpy as np
import cv2
import base64
import time
import imageio
from PIL import Image
import trimesh
import tempfile
import shutil
import glob
import threading
import subprocess
import struct
import zlib
import matplotlib.pyplot as plt
from einops import rearrange
from typing import List, Tuple, Union
try:
    import spaces   
except ImportError:
    # Fallback for local development
    def spaces(func):
        return func
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
import atexit
import uuid
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.predictor import Predictor

from sam2.build_sam import build_sam2_video_predictor
from amodal3r.pipelines import Amodal3RImageTo3DPipeline
from amodal3r.utils import render_utils, postprocessing_utils

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils as render_utils_hi3dgen
from trellis.utils import postprocessing_utils as postprocessing_utils_hi3dgen

from fpose.recover_scale import recover_scale

from oneposeviagen.scripts.estimate_poses import estimate_poses
from oneposeviagen.scripts.render_normals import render_high_model_to_normal_video

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom modules with error handling
try:
    from app_3rd.sam_utils.inference import SamPredictor, get_sam_predictor, run_inference
    from app_3rd.spatrack_utils.infer_track import get_tracker_predictor, run_tracker, get_points_on_a_grid
except ImportError as e:
    logger.error(f"Failed to import custom modules: {e}")
    raise

# Constants
MAX_FRAMES_OFFLINE = 50
MAX_FRAMES_ONLINE = 300
VIDEO_FPS = 10
MAX_SEED = np.iinfo(np.int32).max

COLORS = [(0, 0, 255), (0, 255, 255)]  # BGR: Red for negative, Yellow for positive
MARKERS = [1, 5]  # Cross for negative, Star for positive
MARKER_SIZE = 8

# Thread pool for delayed deletion
thread_pool_executor = ThreadPoolExecutor(max_workers=2)

def delete_later(path: Union[str, os.PathLike], delay: int = 600):
    """Delete file or directory after specified delay (default 10 minutes)"""
    def _delete():
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"Failed to delete {path}: {e}")
    
    def _wait_and_delete():
        time.sleep(delay)
        _delete()
    
    thread_pool_executor.submit(_wait_and_delete)
    atexit.register(_delete)

def create_user_temp_dir():
    """Create a unique temporary directory for each user session"""
    session_id = str(uuid.uuid4())[:8]  # Short unique ID
    temp_dir = os.path.join("temp_local", f"session_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    return temp_dir

from huggingface_hub import hf_hub_download

vggt4track_model = VGGT4Track.from_pretrained("checkpoints/OnePoseViaGen/SpatialTrackerV2/vggt_front")
vggt4track_model.eval()
vggt4track_model = vggt4track_model.to("cuda")

# Global model initialization
print("🚀 Initializing local SpatialTrackerV2 models...")
tracker_model_offline = Predictor.from_pretrained("checkpoints/OnePoseViaGen/SpatialTrackerV2/tracker_offline")
tracker_model_offline.eval()
tracker_model_online = Predictor.from_pretrained("checkpoints/OnePoseViaGen/SpatialTrackerV2/tracker_online")
tracker_model_online.eval() 
predictor = get_sam_predictor()
print("✅ SpatialTrackerV2 models loaded successfully!")

print("🚀 Initializing Amodal3R models...")
trellis_pipeline = Amodal3RImageTo3DPipeline.from_pretrained("checkpoints/OnePoseViaGen/Amodal3R")
trellis_pipeline.cuda()
print("✅ Amodal3R models loaded successfully!")

print("🚀 Initializing Hi3dGen_Color models...")
hi3dgen_pipeline = TrellisImageTo3DPipeline.from_pretrained("checkpoints/OnePoseViaGen/Hi3DGen_Color")
hi3dgen_pipeline.cuda()
print("✅ Hi3dGen_Color models loaded successfully!")

gr.set_static_paths(paths=[Path.cwd().absolute()/"_viz"]) 

# @spaces.GPU
def gpu_run_inference(predictor_arg, image, points, boxes):
    """GPU-accelerated SAM inference"""
    if predictor_arg is None:
        print("Initializing SAM predictor inside GPU function...")
        predictor_arg = get_sam_predictor(predictor=predictor)
    
    # Ensure predictor is on GPU
    try:
        if hasattr(predictor_arg, 'model'):
            predictor_arg.model = predictor_arg.model.cuda()
        elif hasattr(predictor_arg, 'sam'):
            predictor_arg.sam = predictor_arg.sam.cuda()
        elif hasattr(predictor_arg, 'to'):
            predictor_arg = predictor_arg.to('cuda')
        
        if hasattr(image, 'cuda'):
            image = image.cuda()
            
    except Exception as e:
        print(f"Warning: Could not move predictor to GPU: {e}")
    
    return run_inference(predictor_arg, image, points, boxes)

def load_video_with_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV 默认读取为 BGR 格式，转为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # 转为 Tensor: (T, H, W, C) -> (T, C, H, W)
    video_array = np.stack(frames)
    video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)

    return video_tensor

def process_tensor(video_path, fps):
    # import decord
    import torchvision.transforms as T
    video_tensor = load_video_with_opencv(video_path)
    # Resize to ensure minimum side is 336
    h, w = video_tensor.shape[2:]
    scale = 336 / min(h, w)
    if scale < 1:
        new_h, new_w = int(h * scale), int(w * scale)
        video_tensor = T.Resize((new_h, new_w))(video_tensor)
    
    video_tensor = video_tensor[::fps].float()[:MAX_FRAMES_OFFLINE]
    
    # Move to GPU
    video_tensor = video_tensor.cuda()
    print(f"Video tensor shape: {video_tensor.shape}, device: {video_tensor.device}")

    # run vggt 
    # process the image tensor
    video_tensor = preprocess_image(video_tensor)[None]
    return video_tensor

def process_and_save_rgb(video_path, user_temp_dir, fps):
    from torchvision.utils import save_image
    video_tensor = process_tensor(video_path, fps)
    result = video_tensor[0]
    T_img = result.shape[0]
    output_dir = os.path.join(user_temp_dir, 'rgb')
    os.makedirs(output_dir, exist_ok=True)
    for i in range(T_img):
        img_tensor = result[i]
        filename = f"{i:06d}.jpg"
        filepath = os.path.join(output_dir, filename)

        # Normalize if needed (assuming input is in [0, 1])
        if img_tensor.max() <= 1.0:
            img_tensor = img_tensor.clamp(0, 1)
        else:
            img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

        # Save using torchvision's save_image
        save_image(img_tensor, filepath)

# @spaces.GPU
def gpu_run_tracker(tracker_model_arg, tracker_viser_arg, temp_dir, video_name, grid_size, vo_points, fps, mode="offline"):
    """GPU-accelerated tracking"""
    import torchvision.transforms as T
    # import decord
    
    if tracker_model_arg is None or tracker_viser_arg is None:
        print("Initializing tracker models inside GPU function...")
        out_dir = os.path.join(temp_dir, "results")
        os.makedirs(out_dir, exist_ok=True) 
        if mode == "offline":
            tracker_model_arg, tracker_viser_arg = get_tracker_predictor(out_dir, vo_points=vo_points,
                                                                         tracker_model=tracker_model_offline.cuda())
        else:
            tracker_model_arg, tracker_viser_arg = get_tracker_predictor(out_dir, vo_points=vo_points,
                                                                         tracker_model=tracker_model_online.cuda())
    
    # Setup paths
    video_path = os.path.join(temp_dir, f"{video_name}.mp4")
    mask_path = os.path.join(temp_dir, f"{video_name}.png")
    out_dir = os.path.join(temp_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    
    video_tensor = process_tensor(video_path, fps)

    depth_tensor = None
    intrs = None
    extrs = None
    data_npz_load = {}

    # run vggt 
    # process the image tensor
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = vggt4track_model(video_tensor.cuda()/255)
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]

    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    video_tensor = video_tensor.squeeze()
    #NOTE: 20% of the depth is not reliable
    # threshold = depth_conf.squeeze()[0].view(-1).quantile(0.6).item()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
    # Load and process mask
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (video_tensor.shape[3], video_tensor.shape[2]))
        mask = mask.sum(axis=-1)>0
    else:
        mask = np.ones_like(video_tensor[0,0].cpu().numpy())>0
        grid_size = 10

    # Get frame dimensions and create grid points
    frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cuda")
    
    # Sample mask values at grid points and filter
    if os.path.exists(mask_path):
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int.cpu()[...,1], grid_pts_int.cpu()[...,0]]
        grid_pts = grid_pts[:, mask_values]
    
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].cpu().numpy()
    print(f"Query points shape: {query_xyt.shape}")
    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = tracker_model_arg.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2)

        # Resize results to avoid large I/O
        max_size = 518
        h, w = video.shape[2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            video = T.Resize((new_h, new_w))(video)
            video_tensor = T.Resize((new_h, new_w))(video_tensor)
            point_map = T.Resize((new_h, new_w))(point_map)
            track2d_pred[...,:2] = track2d_pred[...,:2] * scale
            intrs[:,:2,:] = intrs[:,:2,:] * scale
            conf_depth = T.Resize((new_h, new_w))(conf_depth)
        
        # Visualize tracks
        tracker_viser_arg.visualize(video=video[None],
                        tracks=track2d_pred[None][...,:2],
                        visibility=vis_pred[None],filename="test")
                        
        # Save in tapip3d format
        data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3].cpu(), track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:].cpu()).numpy()
        data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_npz_load["intrinsics"] = intrs.cpu().numpy()
        data_npz_load["depths"] = point_map[:,2,...].cpu().numpy()
        data_npz_load["video"] = (video_tensor).cpu().numpy()/255
        data_npz_load["visibs"] = vis_pred.cpu().numpy()
        data_npz_load["confs"] = conf_pred.cpu().numpy()
        data_npz_load["confs_depth"] = conf_depth.cpu().numpy()

        depth_names = []
        output_path = os.path.join(temp_dir, "depth")
        os.makedirs(output_path, exist_ok=True)
        for frame_id, depth_map_save in enumerate(data_npz_load["depths"]):
            depth_map_mm = (depth_map_save * 1000).astype('uint16')
            depth_path = f"{output_path}/{frame_id:06d}.png"
            cv2.imwrite(depth_path, depth_map_mm)
            depth_names.append(depth_path)

        intrinsic_file_path = os.path.join(temp_dir, 'intrinsics.json')
        intrinsics_dict = {
        str(frame_id): data_npz_load["intrinsics"][frame_id].tolist()  # 转为 list 才能被 json 序列化
        for frame_id in range(len(data_npz_load["intrinsics"]))
        }
        with open(intrinsic_file_path, 'w') as f:
            json.dump(intrinsics_dict, f, indent=2)
        
        np.savez(os.path.join(out_dir, f'result.npz'), **data_npz_load)
            
    return depth_names

def compress_and_write(filename, header, blob):
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    with open(filename, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(blob)

def process_point_cloud_data(npz_file, width=256, height=192, fps=4):
    fixed_size = (width, height)
    
    data = np.load(npz_file)
    extrinsics = data["extrinsics"]
    intrinsics = data["intrinsics"]
    trajs = data["coords"]
    T, C, H, W = data["video"].shape
    
    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 1, 1]
    fov_y = 2 * np.arctan(H / (2 * fy)) * (180 / np.pi)
    fov_x = 2 * np.arctan(W / (2 * fx)) * (180 / np.pi)
    original_aspect_ratio = (W / fx) / (H / fy)
    
    rgb_video = (rearrange(data["video"], "T C H W -> T H W C") * 255).astype(np.uint8)
    rgb_video = np.stack([cv2.resize(frame, fixed_size, interpolation=cv2.INTER_AREA)
                          for frame in rgb_video])
    
    depth_video = data["depths"].astype(np.float32)
    if "confs_depth" in data.keys():
        confs = (data["confs_depth"].astype(np.float32) > 0.5).astype(np.float32)
        depth_video = depth_video * confs
    depth_video = np.stack([cv2.resize(frame, fixed_size, interpolation=cv2.INTER_NEAREST)
                            for frame in depth_video])
    
    scale_x = fixed_size[0] / W
    scale_y = fixed_size[1] / H
    intrinsics = intrinsics.copy()
    intrinsics[:, 0, :] *= scale_x
    intrinsics[:, 1, :] *= scale_y
    
    min_depth = float(depth_video.min()) * 0.8
    max_depth = float(depth_video.max()) * 1.5
    
    depth_normalized = (depth_video - min_depth) / (max_depth - min_depth)
    depth_int = (depth_normalized * ((1 << 16) - 1)).astype(np.uint16)
    
    depths_rgb = np.zeros((T, fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
    depths_rgb[:, :, :, 0] = (depth_int & 0xFF).astype(np.uint8)
    depths_rgb[:, :, :, 1] = ((depth_int >> 8) & 0xFF).astype(np.uint8)
    
    first_frame_inv = np.linalg.inv(extrinsics[0])
    normalized_extrinsics = np.array([first_frame_inv @ ext for ext in extrinsics])
    
    normalized_trajs = np.zeros_like(trajs)
    for t in range(T):
        homogeneous_trajs = np.concatenate([trajs[t], np.ones((trajs.shape[1], 1))], axis=1)
        transformed_trajs = (first_frame_inv @ homogeneous_trajs.T).T
        normalized_trajs[t] = transformed_trajs[:, :3]
    
    arrays = {
        "rgb_video": rgb_video,
        "depths_rgb": depths_rgb,
        "intrinsics": intrinsics,
        "extrinsics": normalized_extrinsics,
        "inv_extrinsics": np.linalg.inv(normalized_extrinsics),
        "trajectories": normalized_trajs.astype(np.float32),
        "cameraZ": 0.0
    }
    
    header = {}
    blob_parts = []
    offset = 0
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        arr_bytes = arr.tobytes()
        header[key] = {
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "offset": offset,
            "length": len(arr_bytes)
        }
        blob_parts.append(arr_bytes)
        offset += len(arr_bytes)
    
    raw_blob = b"".join(blob_parts)
    compressed_blob = zlib.compress(raw_blob, level=9)
    
    header["meta"] = {
        "depthRange": [min_depth, max_depth],
        "totalFrames": int(T),
        "resolution": fixed_size,
        "baseFrameRate": fps,
        "numTrajectoryPoints": normalized_trajs.shape[1],
        "fov": float(fov_y),
        "fov_x": float(fov_x),
        "original_aspect_ratio": float(original_aspect_ratio),
        "fixed_aspect_ratio": float(fixed_size[0]/fixed_size[1])
    }
    
    compress_and_write('./_viz/data.bin', header, compressed_blob)
    with open('./_viz/data.bin', "rb") as f:
        encoded_blob = base64.b64encode(f.read()).decode("ascii")
    os.unlink('./_viz/data.bin')
    
    random_path = f'./_viz/_{time.time()}.html'
    with open('./_viz/viz_template.html') as f:
        html_template = f.read()
    html_out = html_template.replace(
        "<head>",
        f"<head>\n<script>window.embeddedBase64 = `{encoded_blob}`;</script>"
    )
    with open(random_path,'w') as f:
        f.write(html_out)
    
    return random_path 

def numpy_to_base64(arr):
    """Convert numpy array to base64 string"""
    return base64.b64encode(arr.tobytes()).decode('utf-8')

def base64_to_numpy(b64_str, shape, dtype):
    """Convert base64 string back to numpy array"""
    return np.frombuffer(base64.b64decode(b64_str), dtype=dtype).reshape(shape)

def get_video_name(video_path):
    """Extract video name without extension"""
    return os.path.splitext(os.path.basename(video_path))[0]

def extract_first_frame(video_path):
    """Extract first frame from video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            return None
    except Exception as e:
        print(f"Error extracting first frame: {e}")
        return None

def initialize_predictor(checkpoint):
    """Initialize the SAM2 video predictor with the specified checkpoint."""
    global predictor_sam
    if checkpoint == "tiny":
        sam2_checkpoint = "checkpoints/OnePoseViaGen/SAM2/sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"
    elif checkpoint == "small":
        sam2_checkpoint = "checkpoints/OnePoseViaGen/SAM2/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
    elif checkpoint == "base-plus":
        sam2_checkpoint = "checkpoints/OnePoseViaGen/SAM2/sam2_hiera_base_plus.pt"
        model_cfg = "sam2_hiera_b+.yaml"
    elif checkpoint == "large":
        sam2_checkpoint = "checkpoints/OnePoseViaGen/SAM2/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
    else:
        raise ValueError("Invalid checkpoint")

    predictor_sam = build_sam2_video_predictor(model_cfg, sam2_checkpoint)


def handle_video_upload(video, fps):
    """Handle video upload and extract first frame"""
    if video is None:
        return (None, None, [], 
                gr.update(value=50), 
                gr.update(value=756), 
                gr.update(value=3), {})
    
    # Create user-specific temporary directory
    user_temp_dir = create_user_temp_dir()
    
    # Get original video name and copy to temp directory
    if isinstance(video, str):
        video_name = get_video_name(video)
        video_path = os.path.join(user_temp_dir, f"{video_name}.mp4")
        shutil.copy(video, video_path)
    else:
        video_name = get_video_name(video.name)
        video_path = os.path.join(user_temp_dir, f"{video_name}.mp4")
        with open(video_path, 'wb') as f:
            f.write(video.read())

    print(f"📁 Video saved to: {video_path}")
    
    # Extract first frame
    frame = extract_first_frame(video_path)
    if frame is None:
        return (None, None, [], 
                gr.update(value=50), 
                gr.update(value=756), 
                gr.update(value=3))
    
    # Resize frame to have minimum side length of 336
    h, w = frame.shape[:2]
    scale = 336 / min(h, w)
    new_h, new_w = int(h * scale)//2*2, int(w * scale)//2*2
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Store frame data with temp directory info
    frame_data = {
        'data': numpy_to_base64(frame),
        'shape': frame.shape,
        'dtype': str(frame.dtype),
        'temp_dir': user_temp_dir,
        'video_name': video_name,
        'video_path': video_path
    }

    process_and_save_rgb(video_path, user_temp_dir, fps)
    
    # Get video-specific settings
    print(f"🎬 Video path: '{video}' -> Video name: '{video_name}'")
    grid_size_val, vo_points_val, fps_val = get_video_settings(video_name)
    print(f"🎬 Video settings for '{video_name}': grid_size={grid_size_val}, vo_points={vo_points_val}, fps={fps_val}")

    return (json.dumps(frame_data), frame, [], 
            gr.update(value=grid_size_val), 
            gr.update(value=vo_points_val), 
            gr.update(value=fps), {})

def save_masks(o_masks, video_name, temp_dir):
    """Save binary masks to files in user-specific temp directory"""
    o_files = []
    for mask, _ in o_masks:
        o_mask = np.uint8(mask.squeeze() * 255)
        o_file = os.path.join(temp_dir, f"{video_name}.png")
        cv2.imwrite(o_file, o_mask)
        o_files.append(o_file)
    return o_files

def select_point(original_img: str, sel_pix: list, evt: gr.SelectData, objects):
    """Handle point selection for SAM"""
    point_type = 'positive_point'
    if original_img is None:
        return None, []
    
    try:
        # Convert stored image data back to numpy array
        frame_data = json.loads(original_img)
        original_img_array = base64_to_numpy(frame_data['data'], frame_data['shape'], frame_data['dtype'])
        temp_dir = frame_data.get('temp_dir', 'temp_local')
        video_name = frame_data.get('video_name', 'video')
        
        # Create a display image for visualization
        display_img = original_img_array.copy()
        new_sel_pix = sel_pix.copy() if sel_pix else []
        new_sel_pix.append((evt.index, 1 if point_type == 'positive_point' else 0))
        
        print(f"🎯 Running SAM inference for point: {evt.index}, type: {point_type}")
        # Run SAM inference
        o_masks = gpu_run_inference(None, original_img_array, new_sel_pix, [])
        
        # Draw points on display image
        for point, label in new_sel_pix:
            cv2.drawMarker(display_img, point, COLORS[label], markerType=MARKERS[label], markerSize=MARKER_SIZE, thickness=2)
        
        # Draw mask overlay on display image
        if o_masks:
            mask = o_masks[0][0]
            overlay = display_img.copy()
            overlay[mask.squeeze()!=0] = [20, 60, 200]  # Light blue
            display_img = cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0)
            
            # Save mask for tracking
            save_masks(o_masks, video_name, temp_dir)
            print(f"✅ Mask saved for video: {video_name}")
        
        object_id = 1
        x, y = evt.index[0], evt.index[1]
        if object_id not in objects:
            objects[object_id] = {"points": [], "mask": None, "color": plt.get_cmap("tab10")(len(objects) % 10)[:3]}
        objects[object_id]["points"].append((x, y, point_type))
        objects[object_id]["mask"] = mask > 0.0

        return display_img, new_sel_pix, objects
        
    except Exception as e:
        print(f"❌ Error in select_point: {e}")
        return None, [], {}
    
def transform_point(point, original_size, target_size=518, mode="crop", keep_ratio=False):
    """
    Transform point coordinates based on the image preprocessing applied by preprocess_image.
    
    Args:
        point (tuple): Original point coordinates (x, y)
        original_size (tuple): Original image size (H, W)
        target_size (int): Target size for width/height in preprocess_image
        mode (str): 'crop' or 'pad'
        keep_ratio (bool): Whether to keep aspect ratio when cropping
        
    Returns:
        tuple: Transformed point coordinates (x', y')
    """
    H, W = original_size
    x, y = point
    
    if mode == "pad":
        # Calculate new dimensions after padding
        if W >= H:
            new_W = target_size
            new_H = round(H * (new_W / W) / 14) * 14
        else:
            new_H = target_size
            new_W = round(W * (new_H / H) / 14) * 14
            
        # Calculate scale factors
        scale_x = new_W / W
        scale_y = new_H / H
        
        # Apply scaling
        x_new, y_new = x * scale_x, y * scale_y
        
        # Calculate padding and adjust coordinates accordingly
        h_padding = target_size - new_H
        w_padding = target_size - new_W
        pad_top = h_padding // 2
        pad_left = w_padding // 2
        
        return x_new + pad_left, y_new + pad_top
    
    elif mode == "crop":
        # Calculate new dimensions after cropping
        new_W = target_size
        new_H = round(H * (new_W / W) / 14) * 14
        
        # Calculate scale factors
        scale_x = new_W / W
        scale_y = new_H / H
        
        # Apply scaling
        x_new, y_new = x * scale_x, y * scale_y
        
        # If keep_ratio is False and height exceeds target size, adjust y coordinate
        if not keep_ratio and new_H > target_size:
            start_y = (new_H - target_size) // 2
            y_new -= start_y
        
        return x_new, y_new

def segment_video(objects, original_image_state, fps=VIDEO_FPS):
    """Segment the entire video based on the annotated points."""
    frame_data = json.loads(original_image_state)
    temp_dir = frame_data.get('temp_dir', 'temp_local')
    rgb_dir = os.path.join(temp_dir, "rgb")
    frame_names = sorted([p for p in os.listdir(rgb_dir) if p.endswith('.jpg')])

    inference_state = predictor_sam.init_state(video_path=rgb_dir)
    predictor_sam.reset_state(inference_state)

    # Initial annotation for each object
    for obj_id, obj_data in objects.items():
        obj_data["new_points"] = []
        for point in obj_data["points"]:
            new_x, new_y = transform_point((point[0], point[1]), (obj_data["mask"][0][0].shape))
            obj_data["new_points"].append((int(new_x), int(new_y), point[2]))
        np_points = np.array([[p[0], p[1]] for p in obj_data["new_points"]], dtype=np.float32)
        labels = np.array([1 if p[2] == "positive_point" else 0 for p in obj_data["new_points"]], dtype=np.int32)

        predictor_sam.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            points=np_points,
            labels=labels,
        )

    obj_mask_dir = os.path.join(temp_dir, f"masks")
    os.makedirs(obj_mask_dir, exist_ok=True)
    video_dir = obj_mask_dir
    output_video_path = os.path.join(video_dir, "output_video.mp4")
    extracted_video_paths = {}

    first_frame = cv2.imread(os.path.join(rgb_dir, frame_names[0]))
    height, width = first_frame.shape[:2]

    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    object_writers = {}

    for obj_id in objects.keys():
        extracted_video_paths[obj_id] = os.path.join(video_dir, f"extracted_video_obj_{obj_id}.mp4")
        object_writers[obj_id] = cv2.VideoWriter(extracted_video_paths[obj_id], cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor_sam.propagate_in_video(inference_state):
        frame = cv2.imread(os.path.join(rgb_dir, frame_names[out_frame_idx]))
        overlay_frame = frame.copy()

        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()  # boolean mask
            color = np.array(objects[out_obj_id]["color"]) * 255

            # === 新增：保存为黑白二值图 ===
            binary_mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # 黑色背景
            binary_mask[mask] = 255  # 白色前景（物体）

            # 保存为 PNG 格式，保留透明通道或仅灰度
            mask_filename = os.path.join(obj_mask_dir, f"{out_frame_idx:06d}.png")
            cv2.imwrite(mask_filename, binary_mask)

            # For output video with overlay
            overlay_frame[mask] = overlay_frame[mask] * 0.5 + color * 0.5

            # For individual object videos
            object_frame = np.zeros_like(frame)
            object_frame[mask] = frame[mask]
            object_writers[out_obj_id].write(object_frame)

        video_writer.write(overlay_frame)

    video_writer.release()
    for writer in object_writers.values():
        writer.release()

    output_video_path_new = os.path.join(video_dir, "output_video_new.mp4")
    convert_video_to_mp4(output_video_path, output_video_path_new)
    os.remove(output_video_path)

    return output_video_path_new

def convert_video_to_mp4(input_path, output_path):
    """Convert video to MP4 format using ffmpeg."""
    command = [
        '/usr/bin/ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-y',
        output_path
    ]
    subprocess.run(command, check=True)

def reset_points(original_img: str, sel_pix):
    """Reset all points and clear the mask"""
    if original_img is None:
        return None, []
    
    try:
        # Convert stored image data back to numpy array
        frame_data = json.loads(original_img)
        original_img_array = base64_to_numpy(frame_data['data'], frame_data['shape'], frame_data['dtype'])
        temp_dir = frame_data.get('temp_dir', 'temp_local')
        
        # Create a display image (just the original image)
        display_img = original_img_array.copy()
        
        # Clear all points
        new_sel_pix = []
        
        # Clear any existing masks
        for mask_file in glob.glob(os.path.join(temp_dir, "*.png")):
            try:
                os.remove(mask_file)
            except Exception as e:
                logger.warning(f"Failed to remove mask file {mask_file}: {e}")
        
        print("🔄 Points and masks reset")
        return display_img, new_sel_pix, {}
        
    except Exception as e:
        print(f"❌ Error in reset_points: {e}")
        return None, [], {}
    
def estimate_depth_intrinsic(grid_size, vo_points, fps, original_image_state, processing_mode):
    """Launch visualization with user-specific temp directory"""
    if original_image_state is None:
        return None
    
    try:
        # Get user's temp directory from stored frame data
        frame_data = json.loads(original_image_state)
        temp_dir = frame_data.get('temp_dir', 'temp_local')
        video_name = frame_data.get('video_name', 'video')
        
        print(f"🚀 Starting tracking for video: {video_name}")
        print(f"📊 Parameters: grid_size={grid_size}, vo_points={vo_points}, fps={fps}, mode={processing_mode}")
        
        # Check for mask files
        video_files = glob.glob(os.path.join(temp_dir, "*.mp4"))
        
        if not video_files:
            print("❌ No video file found")
            return "❌ Error: No video file found", None, None
        
        # Run tracker
        print(f"🎯 Running tracker in {processing_mode} mode...")
        out_dir = os.path.join(temp_dir, "results")
        os.makedirs(out_dir, exist_ok=True)
        
        depth_names = gpu_run_tracker(None, None, temp_dir, video_name, grid_size, vo_points, fps, mode=processing_mode)
        depth_dir = os.path.join(temp_dir, 'depth')
        depth_video_path = os.path.join(depth_dir, 'depth.mp4')   
        depth_video_path_new = os.path.join(depth_dir, 'depth_new.mp4')
        convert_depth_images_to_video(depth_names, depth_video_path, fps=VIDEO_FPS)
        convert_video_to_mp4(depth_video_path, depth_video_path_new)
        os.remove(depth_video_path)

        return depth_video_path_new
    
    except Exception as e:
        print(f"❌ Error in estimate_depth_intrinsic: {e}")
        return None

def convert_depth_images_to_video(file_paths, output_video_path, fps=30):
    """
    将给定路径列表中的单通道深度图归一化到 0~255 并转为三通道，组成视频保存
    
    :param file_paths: 包含所有深度图文件路径的列表
    :param output_video_path: 输出视频文件的路径
    :param fps: 输出视频的帧率
    """

    # 检查文件是否存在
    if not file_paths:
        raise ValueError("file_paths 为空，请检查输入路径")

    # 假设所有图像大小相同，读取第一张图像获取尺寸
    first_image = cv2.imread(file_paths[0], cv2.IMREAD_UNCHANGED)
    if first_image is None:
        raise ValueError("Error loading the first image.")

    height, width = first_image.shape

    # 定义视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

    # 获取全局最小最大值（可选），或逐帧归一化
    # 如果你希望每帧独立归一化，就注释掉下面两行并放在循环内
    all_depths = []
    for file in file_paths:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is not None:
            all_depths.append(img)
    min_val = min(np.min(d) for d in all_depths)
    max_val = max(np.max(d) for d in all_depths)

    print(f"Global min/max depth values: {min_val}, {max_val}")

    # 遍历所有图像文件路径
    for depth_image in all_depths:  # 可以避免重复加载
        # 归一化到 0~255
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 转换为三通道图像
        depth_image_3ch = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

        # 写入视频帧
        video_writer.write(depth_image_3ch)

    # 释放视频写入器
    video_writer.release()

def launch_viz(original_image_state):
    """Launch visualization with user-specific temp directory"""
    if original_image_state is None:
        return None, None, None
    
    try:
        # Get user's temp directory from stored frame data
        frame_data = json.loads(original_image_state)
        temp_dir = frame_data.get('temp_dir', 'temp_local')

        out_dir = os.path.join(temp_dir, "results")
        os.makedirs(out_dir, exist_ok=True)

        # Process results
        npz_path = os.path.join(out_dir, "result.npz")
        track2d_video = os.path.join(out_dir, "test_pred_track.mp4")
        
        if os.path.exists(npz_path):
            print("📊 Processing 6D visualization...")
            html_path = process_point_cloud_data(npz_path)
            
            # Create iframe HTML
            iframe_html = f"""
            <div style='border: 3px solid #667eea; border-radius: 10px; 
                        background: #f8f9ff; height: 650px; width: 100%;
                        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
                        margin: 0; padding: 0; box-sizing: border-box; overflow: hidden;'>
                <iframe id="viz_iframe" src="/gradio_api/file={html_path}" 
                        width="100%" height="650" frameborder="0" 
                        style="border: none; display: block; width: 100%; height: 650px;
                               margin: 0; padding: 0; border-radius: 7px;">
                </iframe>
            </div>
            """
            
            print("✅ Tracking completed successfully!")
            return iframe_html, track2d_video if os.path.exists(track2d_video) else None, html_path
        else:
            print("❌ Tracking failed - no results generated")
            return "❌ Error: Tracking failed to generate results", None, None
            
    except Exception as e:
        print(f"❌ Error in launch_viz: {e}")
        return f"❌ Error: {str(e)}", None, None

def generate_final_mask(seg_path, depth_path, out_path, area_threshold=100, max_iter=50):
    """
    根据分割掩码和深度图生成最终mask，并保存到指定路径。
    :param seg_path: 分割掩码路径
    :param depth_path: 深度图路径
    :param out_path: 输出文件夹路径
    :param area_threshold: 连通域面积阈值
    :param max_iter: 遮挡物mask膨胀最大迭代次数
    """
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 保证读取原始深度

    # 2. 提取物体区域的最小深度（排除噪声，取90%分位数）
    object_mask = seg > 128
    object_depths = depth[object_mask]
    min_depth = np.percentile(object_depths, 90)  # 90%分位数，排除极小噪声

    # 3. 找到比min_depth更小的区域（即更靠近相机的区域）
    occluder_mask = (depth < min_depth) & (~object_mask)

    # 4. 腐蚀操作，消除噪声
    kernel = np.ones((3, 3), np.uint8)
    occluder_mask = cv2.erode(occluder_mask.astype(np.uint8), kernel, iterations=1)

    # 5. 只保留物体bounding box内的遮挡区域
    x, y, w, h = cv2.boundingRect(object_mask.astype(np.uint8))
    occluder_mask_bbox = np.zeros_like(occluder_mask)
    occluder_mask_bbox[y:y+h, x:x+w] = occluder_mask[y:y+h, x:x+w]

    # 6. 连通域分析，只保留面积大于阈值的遮挡区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(occluder_mask_bbox, connectivity=8)
    filtered_mask = np.zeros_like(occluder_mask_bbox)
    for i in range(1, num_labels):  # 0是背景
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            filtered_mask[labels == i] = 1

    # 7. 填充遮挡物mask与物体mask之间的间隙
    # 方法：膨胀遮挡物mask，直到与物体mask相连
    dilated_occluder = filtered_mask.copy()
    for i in range(max_iter):
        # 判断是否已经相连
        overlap = (dilated_occluder > 0) & (object_mask > 0)
        if np.any(overlap):
            break
        dilated_occluder = cv2.dilate(dilated_occluder, kernel, iterations=1)

    # 只保留膨胀后新增加的部分（即原本的间隙）
    gap_mask = (dilated_occluder > 0) & (~filtered_mask.astype(bool)) & (~object_mask)
    # 将间隙区域也视为遮挡物
    filtered_mask[gap_mask] = 1

    # 8. 生成最终mask：白色255为背景，灰色188为物体，黑色0为遮挡物
    final_mask = np.ones_like(seg, dtype=np.uint8) * 255
    final_mask[object_mask] = 188
    final_mask[filtered_mask > 0] = 0
    
    return Image.fromarray(final_mask)

def generate_model_and_rescale_model(original_image_state, seed, randomize_seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, is_occluded):
    frame_data = json.loads(original_image_state)
    temp_dir = frame_data.get('temp_dir', 'temp_local')
    rgb_dir = os.path.join(temp_dir, 'rgb')
    rgb_names = [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
    rgb_names.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    depth_dir = os.path.join(temp_dir, 'depth')
    depth_names = [os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(".png")]
    depth_names.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    mask_dir = os.path.join(temp_dir, 'masks')
    mask_names = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")]
    mask_names.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    with open(os.path.join(temp_dir, 'intrinsics.json'), 'r') as f:
        intrinsics = json.load(f)
        intrinsic = intrinsics['0']

    model_dir = os.path.join(temp_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    # rgb_image = mask_image(rgb_names[0], mask_names[0])
    # rgb_image = upscale_image_if_needed(rgb_image)
    
    rgb_image = Image.open(rgb_names[0])
    if is_occluded:
        final_mask = generate_final_mask(mask_names[0], depth_names[0], model_dir, area_threshold=100, max_iter=50)
    else:
        # 读取原始mask，黑色变为白色，白色变为灰色（188），输出单通道图
        mask_img = cv2.imread(mask_names[0], cv2.IMREAD_GRAYSCALE)
        final_mask = np.ones_like(mask_img, dtype=np.uint8) * 255  # 全部先设为白色
        final_mask[mask_img > 127] = 188  # 原mask白色区域变为灰色188
        final_mask = Image.fromarray(final_mask)
        rgb_image = mask_image(rgb_names[0], mask_names[0])
    final_mask.save(os.path.join(model_dir, 'final_mask.png'))
    
    seed = get_seed(randomize_seed=randomize_seed, seed=seed)
    video_path, mesh_path, high_mesh_path = generate_3d(rgb_image, final_mask, temp_dir, 'obj', seed, ss_guidance_strength=ss_guidance_strength, ss_sampling_steps=ss_sampling_steps, slat_guidance_strength=slat_guidance_strength, slat_sampling_steps=slat_sampling_steps, is_occluded=is_occluded)

    scaled_model_path, scale, anchor_pose = recover_true_scale(mesh_path, depth_names[0], intrinsic, rgb_names[0], mask_names[0], model_dir)
    high_mesh = trimesh.load(high_mesh_path)
    high_mesh.vertices = high_mesh.vertices * scale
    high_mesh.export(high_mesh_path)

    return video_path, scaled_model_path, high_mesh_path

def mask_image(rgb_path, mask_path) -> Image.Image:
    """
    Preprocess the input image.
    """
    # 将输入图像转换为numpy数组
    input = Image.open(rgb_path)
    input_np = np.array(input)
    
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = input_np[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    
    # 使用alpha通道或者移除背景
    if has_alpha:
        output = input
    else:
        input = input.convert('RGB')
        
        # 假设我们已经有了对应的单通道mask图像
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask_binary = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)
        
        # 应用掩码
        rgb_img = cv2.cvtColor(np.array(input), cv2.COLOR_RGB2BGR)
        result_img = cv2.bitwise_and(rgb_img, rgb_img, mask=mask_binary)
        
        # 转换回PIL Image格式
        result_img_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        output = result_img_pil
        
    # 计算alpha通道或mask的有效区域
    output_np = np.array(output)
    if output.mode == 'RGBA':
        alpha = output_np[:, :, 3]
    else:
        alpha = np.array(mask_binary)
    
    bbox = np.argwhere(alpha > 0.8 * 255)
    if len(bbox) == 0:
        return output
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = output.resize((518, 518), Image.LANCZOS)
    output_np = np.array(output).astype(np.float32) / 255
    if output_np.shape[2] == 4:  # 如果是带alpha通道的图像
        output_np = output_np[:, :, :3] * output_np[:, :, 3:4]
    output = Image.fromarray((output_np * 255).astype(np.uint8))
    return output

def generate_3d(image: Image.Image, mask, temp_dir,
                export_format: str, seed: int = -1,
                ss_guidance_strength: float = 7.5, ss_sampling_steps: int = 12, 
                slat_guidance_strength: float = 15, slat_sampling_steps: int = 25, is_occluded: bool = False):
    """Generate 3D model and preview video from input image."""
    if seed == -1:
        seed = np.random.randint(0, MAX_SEED)
    
    if is_occluded:
        outputs = trellis_pipeline.run_multi_image(
            [image],
            [mask],
            seed=seed,
            formats=["mesh", "gaussian"],
            # preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
        generated_mesh = outputs['mesh'][0]
        generated_gs = outputs['gaussian'][0]
            
        # Save video and mesh
        model_dir = os.path.join(temp_dir, 'model')
        model_middle_dir = os.path.join(model_dir, 'middle_file')
        os.makedirs(model_middle_dir, exist_ok=True)
        output_id = str(uuid.uuid4())
        video_path = f"{model_middle_dir}/{output_id}_preview.mp4"
        mesh_path = f"{model_dir}/model.{export_format}"
        # gs_path = f"{model_middle_dir}/{output_id}.ply"
        # slat_path = f"{model_middle_dir}/{output_id}.npz"
        # generated_slat = outputs['slat'][0]
        
        # save_slat(generated_slat, slat_path)
        # Save video
        video_geo = render_utils.render_video(generated_gs, resolution=1024, num_frames=120)['color']
        imageio.mimsave(video_path, video_geo, fps=15)
        trimesh_mesh = postprocessing_utils.to_glb(generated_gs, generated_mesh, verbose=False)
        
    else:
        outputs = hi3dgen_pipeline.run(
            image,
            seed=seed,
            formats=["mesh", "gaussian"],
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
        generated_mesh = outputs['mesh'][0]
        generated_gs = outputs['gaussian'][0]
            
        # Save video and mesh
        model_dir = os.path.join(temp_dir, 'model')
        model_middle_dir = os.path.join(model_dir, 'middle_file')
        os.makedirs(model_middle_dir, exist_ok=True)
        output_id = str(uuid.uuid4())
        video_path = f"{model_middle_dir}/{output_id}_preview.mp4"
        mesh_path = f"{model_dir}/model.{export_format}"
        # gs_path = f"{model_middle_dir}/{output_id}.ply"
        # slat_path = f"{model_middle_dir}/{output_id}.npz"
        # generated_slat = outputs['slat'][0]
        
        # save_slat(generated_slat, slat_path)
        # Save video
        video_geo = render_utils_hi3dgen.render_video(generated_mesh, resolution=1024, num_frames=120)['color']
        imageio.mimsave(video_path, video_geo, fps=15)
        trimesh_mesh = postprocessing_utils_hi3dgen.to_trimesh(generated_gs, generated_mesh, verbose=False)

    # 绕x轴正方向旋转90度
    R = trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0])
    trimesh_mesh.apply_transform(R)
    trimesh_mesh.export(mesh_path, file_type='obj')

    high_mesh_path = f"{model_middle_dir}/{output_id}_high_mesh.obj"
    save_mesh(generated_mesh, high_mesh_path)
    
    return video_path, mesh_path, high_mesh_path

def save_mesh(mesh_result, filename):
    vertices = mesh_result.vertices.cpu().numpy() if hasattr(mesh_result.vertices, 'cpu') else mesh_result.vertices
    faces = mesh_result.faces.cpu().numpy() if hasattr(mesh_result.faces, 'cpu') else mesh_result.faces
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    if mesh_result.vertex_attrs is not None:
        attrs = mesh_result.vertex_attrs.cpu().numpy() if hasattr(mesh_result.vertex_attrs, 'cpu') else mesh_result.vertex_attrs
        mesh.visual.vertex_colors = attrs
    
    mesh.export(filename)

def recover_true_scale(normal_model_path: str, anchor_depth_name: list, anchor_intrinsic: list, anchor_image_name: str, anchor_mask_name: str, output_dir: str):

    intrinsic_path = os.path.join(output_dir, 'anchor_file')
    os.makedirs(intrinsic_path, exist_ok=True)
    intrinsic_file = os.path.join(intrinsic_path, 'intrinsic.txt')
    np.savetxt(intrinsic_file, anchor_intrinsic, fmt='%.6f')
    mid_dir = os.path.join(output_dir, 'mid_files')
    os.makedirs(mid_dir, exist_ok=True)

    scaled_mesh_path = os.path.join(mid_dir, 'scaled_mesh.obj')

    #recover the true scale of the model from the anchor image
    scaled_mesh, pose, final_scale = recover_scale(normal_model_path, anchor_depth_name, anchor_image_name, anchor_mask_name, intrinsic_file, 'test', mid_dir)
    scaled_mesh.export(scaled_mesh_path)

    return scaled_mesh_path, final_scale, pose

def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

def save_slat(slat, save_path: str):
    """Save SLAT features and coordinates to a npz file."""
    feats_numpy = slat.feats.detach().cpu().numpy()
    coords_numpy = slat.coords.detach().cpu().numpy()

    np.savez(
        save_path,
        feats=feats_numpy,
        coords=coords_numpy
    )

def estimate_query_poses(original_image_state, scaled_model_path, high_mesh_path):
    #estimate the poses of the query images
    frame_data = json.loads(original_image_state)
    temp_dir = frame_data.get('temp_dir', 'temp_local')
    user_dir = os.path.join(temp_dir, 'pose_debug')

    pose_dir = os.path.join(temp_dir, 'pose_result')
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(user_dir, exist_ok=True)

    rgb_dir = os.path.join(temp_dir, 'rgb')
    rgb_names = [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
    rgb_names.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    depth_dir = os.path.join(temp_dir, 'depth')
    depth_names = [os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(".png")]
    depth_names.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    mask_dir = os.path.join(temp_dir, 'masks')
    mask_names = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")]
    mask_names.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    with open(os.path.join(temp_dir, 'intrinsics.json'), 'r') as f:
        intrinsics_dict = json.load(f)

    intrinsics = [intrinsics_dict[str(id)] for id in range(len(rgb_names))]

    out_dir = os.path.join(temp_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    # Process results
    npz_path = os.path.join(out_dir, "result.npz")
    poses = estimate_poses(npz_path, rgb_names, depth_names, mask_names, intrinsics, scaled_model_path, user_dir, debug=0, est_refine_iter=5)

    poses_file_path = os.path.join(pose_dir, 'poses.json')

    poses_dict = {
        str(frame_id): poses[frame_id].tolist()  # 转为 list 才能被 json 序列化
        for frame_id in range(len(poses))
    }

    with open(poses_file_path, 'w') as f:
        json.dump(poses_dict, f, indent=2)

    normal_video_path = os.path.join(pose_dir, 'noraml_video.mp4')
    normal_video_path_new = os.path.join(pose_dir, 'noraml_video_new.mp4')
    render_high_model_to_normal_video(poses, rgb_names, intrinsics, high_mesh_path, normal_video_path, fps=VIDEO_FPS, device='cuda')
    convert_video_to_mp4(normal_video_path, normal_video_path_new)
    os.remove(normal_video_path)
    return normal_video_path_new

def clear_all():
    """Clear all buffers and temporary files"""
    return (None, None, [], 
            gr.update(value=50), 
            gr.update(value=756), 
            gr.update(value=3))

def clear_all_with_download():
    """Clear all buffers including both download components"""
    viz_html = gr.HTML(
                    label="6D Pose Visualization",
                    value="""
                    <div style='border: 3px solid #667eea; border-radius: 10px; 
                                background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%); 
                                text-align: center; height: 650px; display: flex; 
                                flex-direction: column; justify-content: center; align-items: center;
                                box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15);
                                margin: 0; padding: 20px; box-sizing: border-box;'>
                        <div style='font-size: 56px; margin-bottom: 25px;'>🌐</div>
                        <h3 style='color: #667eea; margin-bottom: 18px; font-size: 28px; font-weight: 600;'>
                            6D Pose Visualization
                        </h3>
                        <p style='color: #666; font-size: 18px; line-height: 1.6; max-width: 550px; margin-bottom: 30px;'>
                            Estimate any object in any RGB video without object model
                        </p>
                        <div style='background: rgba(102, 126, 234, 0.1); border-radius: 30px; 
                                    padding: 15px 30px; border: 1px solid rgba(102, 126, 234, 0.2);'>
                            <span style='color: #667eea; font-weight: 600; font-size: 16px;'>
                                ⚡ Powered by OnePoseviaGen
                            </span>
                        </div>
                    </div>
                    """,
                    elem_id="viz_container"
                )
    return (None, None, None, None, None, None, None, None, [], 
            gr.update(value=50), 
            gr.update(value=756), 
            gr.update(value=3),
            gr.update(value="offline"),  # processing_mode
            None,  # tracking_video_download
            None,
            {}, viz_html)  # HTML download component

def get_video_settings(video_name):
    """Get video-specific settings based on video name"""
    video_settings = {
        "running": (50, 512, 2),
        "backpack": (40, 600, 2),
        "kitchen": (60, 800, 3),
        "pillow": (35, 500, 2),
        "handwave": (35, 500, 8),
        "hockey": (45, 700, 2),
        "drifting": (35, 1000, 6),
        "basketball": (45, 1500, 5),
        "ego_teaser": (45, 1200, 10),
        "robot_unitree": (45, 500, 4),
        "robot_3": (35, 400, 5),
        "teleop2": (45, 256, 7),
        "pusht": (45, 256, 10),
        "cinema_0": (45, 356, 5),
        "cinema_1": (45, 756, 3),
        "robot1": (45, 600, 2),
        "robot2": (45, 600, 2),
        "protein": (45, 600, 2),
        "kitchen_egocentric": (45, 600, 2),
        "ball_ke": (50, 600, 3), 
        "groundbox_800": (50, 756, 3),
        "mug": (50, 756, 3), 
    }
    
    return video_settings.get(video_name, (50, 756, 3)) 

# Create the Gradio interface
print("🎨 Creating Gradio interface...")

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="🎯 [OnePoseviaGen](https://github.com/GZWSAMA/OnePoseviaGen)",
    css="""
    footer{
    display:none !important;
    }
    .gradio-container {
        max-width: 1800px !important;
        margin: auto !important;
    }
    .gr-button {
        margin: 3px;
        border: 1px solid #e1e5e9 !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
    }
    .gr-form {
        background: white;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e1e5e9;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    /* 移除 gr.Group 的默认灰色背景 */
    .gr-form {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 8px !important;
    }
    /* 固定3D可视化器尺寸 - 窄边设计 */
    #viz_container {
        height: 650px !important;
        min-height: 650px !important;
        max-height: 650px !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 8px !important;
    }
    #viz_container > div {
        height: 650px !important;
        min-height: 650px !important;
        max-height: 650px !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }
    #viz_container iframe {
        height: 650px !important;
        min-height: 650px !important;
        max-height: 650px !important;
        width: 100% !important;
        border: none !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
        border-radius: 7px !important;
    }
    /* 固定视频上传组件高度 - 窄边设计 */
    .gr-video {
        height: 300px !important;
        min-height: 300px !important;
        max-height: 300px !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }
    .gr-video video {
        height: 260px !important;
        max-height: 260px !important;
        object-fit: contain !important;
        background: #f8f9fa;
        border-radius: 6px !important;
    }
    .gr-video .gr-video-player {
        height: 260px !important;
        max-height: 260px !important;
        border-radius: 6px !important;
    }
    /* Image组件窄边设计 */
    .gr-image {
        border: 1px solid #e1e5e9 !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }
    /* 输出视频组件窄边设计 */
    .gr-video[data-testid*="output"] {
        border: 1px solid #e1e5e9 !important;
        border-radius: 6px !important;
        padding: 3px !important;
    }
    /* 强力移除examples的灰色背景 - 使用更通用的选择器 */
    .horizontal-examples,
    .horizontal-examples > *,
    .horizontal-examples * {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Examples组件水平滚动样式 - 窄边设计 */
    .horizontal-examples [data-testid="examples"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    .horizontal-examples [data-testid="examples"] > div {
        background: transparent !important;
        background-color: transparent !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        scrollbar-width: thin;
        scrollbar-color: #667eea transparent;
        padding: 0 !important;
        margin-top: 8px;
        border: none !important;
    }
    
    .horizontal-examples [data-testid="examples"] table {
        display: flex !important;
        flex-wrap: nowrap !important;
        min-width: max-content !important;
        gap: 12px !important;
        padding: 8px 0;
        background: transparent !important;
        border: none !important;
    }
    
    .horizontal-examples [data-testid="examples"] tbody {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 12px !important;
        background: transparent !important;
    }
    
    .horizontal-examples [data-testid="examples"] tr {
        display: flex !important;
        flex-direction: column !important;
        min-width: 160px !important;
        max-width: 160px !important;
        margin: 0 !important;
        background: white !important;
        border-radius: 8px;
        border: 1px solid #e1e5e9 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
        cursor: pointer;
        overflow: hidden;
    }
    
    .horizontal-examples [data-testid="examples"] tr:hover {
        transform: translateY(-2px);
        box-shadow: 0 3px 8px rgba(102, 126, 234, 0.15);
        border-color: #667eea !important;
    }
    
    .horizontal-examples [data-testid="examples"] td {
        text-align: center !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    
    .horizontal-examples [data-testid="examples"] td:first-child {
        padding: 0 !important;
        background: transparent !important;
    }
    
    .horizontal-examples [data-testid="examples"] video {
        border-radius: 7px 7px 0 0 !important;
        width: 100% !important;
        height: 85px !important;
        object-fit: cover !important;
        background: #f8f9fa !important;
    }
    
    .horizontal-examples [data-testid="examples"] td:last-child {
        font-size: 11px !important;
        font-weight: 500 !important;
        color: #333 !important;
        padding: 6px 10px !important;
        background: #f8f9fa !important;
        border-radius: 0 0 7px 7px;
        border-top: 1px solid #e1e5e9 !important;
    }
    
    /* 滚动条样式 - 更细的设计 */
    .horizontal-examples [data-testid="examples"] > div::-webkit-scrollbar {
        height: 6px;
    }
    .horizontal-examples [data-testid="examples"] > div::-webkit-scrollbar-track {
        background: transparent;
        border-radius: 3px;
    }
    .horizontal-examples [data-testid="examples"] > div::-webkit-scrollbar-thumb {
        background: #c1c7cd;
        border-radius: 3px;
    }
    .horizontal-examples [data-testid="examples"] > div::-webkit-scrollbar-thumb:hover {
        background: #a8b2ba;
    }
    
    /* 按钮组样式优化 */
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: 1px solid #5a6fd8 !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
    }
    
    .gr-button-secondary {
        background: white !important;
        border: 1px solid #e1e5e9 !important;
        color: #666 !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
    }
    
    .gr-button-secondary:hover {
        border-color: #667eea !important;
        color: #667eea !important;
    }
    
    /* 标题和文本优化 */
    .gr-markdown h3 {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
        margin-bottom: 8px !important;
        border-bottom: 1px solid #e1e5e9 !important;
        padding-bottom: 4px !important;
    }
    
    /* Column间距优化 */
    .gr-column {
        padding: 0 8px !important;
    }
    
    .gr-row {
        margin-bottom: 16px !important;
    }
    """
) as demo:
    
    # Add prominent main title
    
    # gr.Markdown("""
    # # ✨ OnePoseviaGen
                
    # Welcome to [OnePoseviaGen](https://github.com/GZWSAMA/OnePoseviaGen)! This interface allows you to estimate 7D pose of any object using our model.
    # For full information, please refer to the [official website](https://gzwsama.github.io/OnePoseviaGen.github.io/).
    # Please cite our paper and give us a star 🌟 if you find this project useful!
    
    # """)
    
    # Status indicator
    
    # Main content area - video upload left, 3D visualization right
    with gr.Row():
        with gr.Column(scale=1):
            # Video upload section
            gr.Markdown("### 📂 Select Video")
            
            # Define video_input here so it can be referenced in examples
            video_input = gr.Video(
                label="Upload Video or Select Example",
                format="mp4",
                height=270,  # Matched height with 3D viz
            )
                
            interactive_frame = gr.Image(
                label="Click to Select Object",
                type="numpy",
                interactive=True,
                height=270,
            )
                
            reset_points_btn = gr.Button("🔄 Reset Selection", variant="secondary", size="sm")
            is_occluded = gr.Checkbox(label="🎭 Object is Partially Occluded", value=False)
        
        with gr.Column(scale=2):
            # 6D Visualization - wider and taller to match left side
            with gr.Group():
                gr.Markdown("### 🌐 6D Pose Visualization")
                viz_html = gr.HTML(
                    label="6D Pose Visualization",
                    value="""
                    <div style='border: 3px solid #667eea; border-radius: 10px; 
                                background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%); 
                                text-align: center; height: 650px; display: flex; 
                                flex-direction: column; justify-content: center; align-items: center;
                                box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15);
                                margin: 0; padding: 20px; box-sizing: border-box;'>
                        <div style='font-size: 56px; margin-bottom: 25px;'>🌐</div>
                        <h3 style='color: #667eea; margin-bottom: 18px; font-size: 28px; font-weight: 600;'>
                            6D Pose Visualization
                        </h3>
                        <p style='color: #666; font-size: 18px; line-height: 1.6; max-width: 550px; margin-bottom: 30px;'>
                            Estimate any object in any RGB video without object model
                        </p>
                        <div style='background: rgba(102, 126, 234, 0.1); border-radius: 30px; 
                                    padding: 15px 30px; border: 1px solid rgba(102, 126, 234, 0.2);'>
                            <span style='color: #667eea; font-weight: 600; font-size: 16px;'>
                                ⚡ Powered by OnePoseviaGen
                            </span>
                        </div>
                    </div>
                    """,
                    elem_id="viz_container"
                )
    
    # Start button section - below video area
    with gr.Row():
        with gr.Column(scale=3):
            launch_btn = gr.Button("🚀 Start Pipeline Now!", variant="primary", size="lg")
        with gr.Column(scale=1):
            clear_all_btn = gr.Button("🗑️ Clear All", variant="secondary", size="sm")
    
    with gr.Row():
        video_output = gr.Video(label="1. Segmented Video", loop=True, height=150)
        depth_output = gr.Video(label="2. Depth Video", loop=True, height=150)
        model_output = gr.Video(label="3. Generated Model", loop=True, height=150)
        pose_output = gr.Video(label="4. Pose Video", loop=True, height=150)

    # Tracking parameters section
    with gr.Row(visible=False):
        gr.Markdown("### ⚙️ Parameters")
    with gr.Row(visible=False):
        # 添加模式选择器
        with gr.Accordion(label="Tracking Settings", open=False):
            with gr.Column(scale=1):
                processing_mode = gr.Radio(
                    choices=["offline", "online"],
                    value="offline",
                    label="Processing Mode",
                    info="Offline: default mode | Online: Sliding Window Mode"
                )
            with gr.Column(scale=1):
                grid_size = gr.Slider(
                    minimum=10, maximum=100, step=10, value=50,
                    label="Grid Size", info="Tracking detail level"
                )
            with gr.Column(scale=1):
                vo_points = gr.Slider(
                    minimum=100, maximum=2000, step=50, value=756,
                    label="VO Points", info="Motion accuracy"
                )
            with gr.Column(scale=1):
                fps = gr.Slider(
                    minimum=1, maximum=20, step=1, value=3,
                    label="FPS", info="Processing speed"
                )
    with gr.Row(visible=False):
        with gr.Accordion(label="Generation Settings", open=False):
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            gr.Markdown("Stage 1: Sparse Structure Generation")
            with gr.Row():
                ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
            gr.Markdown("Stage 2: Structured Latent Generation")
            with gr.Row():
                slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3, step=0.1)
                slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)

    with gr.Row():
        # Traditional examples but with horizontal scroll styling
        gr.Markdown("🎨**Examples:** (scroll horizontally to see all videos)")
    with gr.Row(elem_classes=["horizontal-examples"]):
        # Horizontal video examples with slider
        # gr.HTML("<div style='margin-top: 5px;'></div>")
        gr.Examples(
            examples = [
                f'{folder}/{video}' 
                for folder in ["assets/example_videos"] 
                for video in os.listdir(folder)
            ],
            inputs=[video_input],
            outputs=[video_input],
            fn=None,
            cache_examples=False,
            label="",
            examples_per_page=10  # Show 6 examples per page so they can wrap to multiple rows
        )

    # Downloads section - hidden but still functional for local processing
    with gr.Row(visible=False):
        with gr.Column(scale=1):
            tracking_video_download = gr.File(
                label="📹 Download 2D Tracking Video",
                interactive=False,
                visible=False
            )
        with gr.Column(scale=1):
            html_download = gr.File(
                label="📄 Download 6D Visualization HTML",
                interactive=False,
                visible=False
            )

    # Acknowledgments Section
    gr.HTML("""
    <div style='background: linear-gradient(135deg, #fff8e1 0%, #fffbf0 100%); 
                border-radius: 8px; padding: 20px; margin: 15px 0; 
                box-shadow: 0 2px 8px rgba(255, 193, 7, 0.1);
                border: 1px solid rgba(255, 193, 7, 0.2);'>
        <div style='text-align: center;'>
            <h3 style='color: #5d4037; margin: 0 0 10px 0; font-size: 18px; font-weight: 600;'>
                📚 Acknowledgments
            </h3>
            <p style='color: #5d4037; margin: 0 0 15px 0; font-size: 14px; line-height: 1.5;'>
                Our code is heavily adapted from the following works, and our visualization is built upon SpatialTracker V2.
            </p>
            <p style='color: #5d4037; margin: 0 0 15px 0; font-size: 14px; line-height: 1.5;'>
                We sincerely thank the authors for their outstanding contributions to the computer vision community.
            </p>
            <a href="https://github.com/NVlabs/FoundationPose" target="_blank" 
               style='display: inline-flex; align-items: center; gap: 8px; 
                      background: rgba(255, 193, 7, 0.15); color: #5d4037; 
                      padding: 10px 20px; border-radius: 25px; text-decoration: none; 
                      font-weight: bold; font-size: 14px; border: 1px solid rgba(255, 193, 7, 0.3);
                      transition: all 0.3s ease;'
               onmouseover="this.style.background='rgba(255, 193, 7, 0.25)'; this.style.transform='translateY(-2px)'"
               onmouseout="this.style.background='rgba(255, 193, 7, 0.15)'; this.style.transform='translateY(0)'">
                📚 Visit FoundationPose Repository
            </a>
            <a href="https://github.com/henry123-boy/SpaTrackerV2" target="_blank" 
               style='display: inline-flex; align-items: center; gap: 8px; 
                      background: rgba(255, 193, 7, 0.15); color: #5d4037; 
                      padding: 10px 20px; border-radius: 25px; text-decoration: none; 
                      font-weight: bold; font-size: 14px; border: 1px solid rgba(255, 193, 7, 0.3);
                      transition: all 0.3s ease;'
               onmouseover="this.style.background='rgba(255, 193, 7, 0.25)'; this.style.transform='translateY(-2px)'"
               onmouseout="this.style.background='rgba(255, 193, 7, 0.15)'; this.style.transform='translateY(0)'">
                📚 Visit SpatialTracker V2 Repository
            </a>
            <a href="https://github.com/microsoft/TRELLIS" target="_blank" 
               style='display: inline-flex; align-items: center; gap: 8px; 
                      background: rgba(255, 193, 7, 0.15); color: #5d4037; 
                      padding: 10px 20px; border-radius: 25px; text-decoration: none; 
                      font-weight: bold; font-size: 14px; border: 1px solid rgba(255, 193, 7, 0.3);
                      transition: all 0.3s ease;'
               onmouseover="this.style.background='rgba(255, 193, 7, 0.25)'; this.style.transform='translateY(-2px)'"
               onmouseout="this.style.background='rgba(255, 193, 7, 0.15)'; this.style.transform='translateY(0)'">
                📚 Visit TRELLIS Repository
            </a>
            <a href="https://github.com/Stable-X/Stable3DGen" target="_blank" 
               style='display: inline-flex; align-items: center; gap: 8px; 
                      background: rgba(255, 193, 7, 0.15); color: #5d4037; 
                      padding: 10px 20px; border-radius: 25px; text-decoration: none; 
                      font-weight: bold; font-size: 14px; border: 1px solid rgba(255, 193, 7, 0.3);
                      transition: all 0.3s ease;'
               onmouseover="this.style.background='rgba(255, 193, 7, 0.25)'; this.style.transform='translateY(-2px)'"
               onmouseout="this.style.background='rgba(255, 193, 7, 0.15)'; this.style.transform='translateY(0)'">
                📚 Visit Stable3DGen Repository
            </a>
            <a href="https://github.com/Sm0kyWu/Amodal3R" target="_blank" 
               style='display: inline-flex; align-items: center; gap: 8px; 
                      background: rgba(255, 193, 7, 0.15); color: #5d4037; 
                      padding: 10px 20px; border-radius: 25px; text-decoration: none; 
                      font-weight: bold; font-size: 14px; border: 1px solid rgba(255, 193, 7, 0.3);
                      transition: all 0.3s ease;'
               onmouseover="this.style.background='rgba(255, 193, 7, 0.25)'; this.style.transform='translateY(-2px)'"
               onmouseout="this.style.background='rgba(255, 193, 7, 0.15)'; this.style.transform='translateY(0)'">
                📚 Visit Amodal3R Repository
            </a>
            <a href="https://github.com/facebookresearch/sam2" target="_blank" 
               style='display: inline-flex; align-items: center; gap: 8px; 
                      background: rgba(255, 193, 7, 0.15); color: #5d4037; 
                      padding: 10px 20px; border-radius: 25px; text-decoration: none; 
                      font-weight: bold; font-size: 14px; border: 1px solid rgba(255, 193, 7, 0.3);
                      transition: all 0.3s ease;'
               onmouseover="this.style.background='rgba(255, 193, 7, 0.25)'; this.style.transform='translateY(-2px)'"
               onmouseout="this.style.background='rgba(255, 193, 7, 0.15)'; this.style.transform='translateY(0)'">
                📚 Visit SAM-2 Repository
            </a>
        </div>
    </div>
    """)

    # Footer
    gr.HTML("""
    <div style='text-align: center; margin: 20px 0 10px 0;'>
        <span style='font-size: 12px; color: #888; font-style: italic;'>
            Powered by OnePoseviaGen | Built with ❤️ for the Computer Vision Community
        </span>
    </div>
    """)

    # Hidden state variables
    original_image_state = gr.State(None)
    selected_points = gr.State([])
    objects = gr.State({})
    scaled_model_path = gr.State(None)
    high_mesh_path = gr.State(None)
    
    # Event handlers
    video_input.change(
        fn=handle_video_upload,
        inputs=[video_input, fps],
        outputs=[original_image_state, interactive_frame, selected_points, grid_size, vo_points, fps, objects]
    )
        
    interactive_frame.select(
        fn=select_point,
        inputs=[original_image_state, selected_points, objects],
        outputs=[interactive_frame, selected_points, objects]
    )
    
    reset_points_btn.click(
        fn=reset_points,
        inputs=[original_image_state, selected_points],
        outputs=[interactive_frame, selected_points, objects]
    )
    
    clear_all_btn.click(
        fn=clear_all_with_download,
        outputs=[original_image_state, video_input, video_output, depth_output, model_output, pose_output, scaled_model_path, interactive_frame, selected_points, grid_size, vo_points, fps, processing_mode, tracking_video_download, html_download, objects, viz_html]
    )
    
    launch_btn.click(
        fn=segment_video,
        inputs=[objects, original_image_state],
        outputs=[video_output]
    ).then(
        fn=estimate_depth_intrinsic,
        inputs=[grid_size, vo_points, fps, original_image_state, processing_mode],
        outputs=[depth_output]
    ).then(
        fn=generate_model_and_rescale_model,
        inputs=[original_image_state, seed, randomize_seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, is_occluded],
        outputs=[model_output, scaled_model_path, high_mesh_path],
    ).then(
        fn=estimate_query_poses,
        inputs=[original_image_state, scaled_model_path, high_mesh_path],
        outputs=[pose_output]
    ).then(
        fn=launch_viz,
        inputs=[original_image_state],
        outputs=[viz_html, tracking_video_download, html_download]
    )

# Launch the interface
if __name__ == "__main__":
    print("🌟 Launching OnePoseviaGen Local Version...")
    initialize_predictor("large")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=3343,
        share=True,
        debug=True,
        show_error=True,
        show_api=False,   # 隐藏API信息
    ) 