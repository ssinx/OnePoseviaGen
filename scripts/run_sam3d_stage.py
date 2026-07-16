import argparse
import importlib.util
import os
import sys

os.environ.setdefault("CONDA_PREFIX", sys.prefix)

import imageio
import numpy as np
import trimesh
from PIL import Image


def load_sam3d_notebook_api(sam3d_root):
    notebook_dir = os.path.join(sam3d_root, "notebook")
    if notebook_dir not in sys.path:
        sys.path.insert(0, notebook_dir)
    if sam3d_root not in sys.path:
        sys.path.insert(0, sam3d_root)

    inference_path = os.path.join(notebook_dir, "inference.py")
    spec = importlib.util.spec_from_file_location("sam3d_notebook_inference", inference_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Inference, module.ready_gaussian_for_video_rendering, module.render_video


def save_mesh(mesh_result, filename):
    vertices = mesh_result.vertices.cpu().numpy() if hasattr(mesh_result.vertices, "cpu") else mesh_result.vertices
    faces = mesh_result.faces.cpu().numpy() if hasattr(mesh_result.faces, "cpu") else mesh_result.faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh_result.vertex_attrs is not None:
        attrs = mesh_result.vertex_attrs.cpu().numpy() if hasattr(mesh_result.vertex_attrs, "cpu") else mesh_result.vertex_attrs
        mesh.visual.vertex_colors = attrs
    mesh.export(filename)


def export_sam3d_mesh(outputs, mesh_path, high_mesh_path):
    mesh_source = outputs.get("glb")
    if mesh_source is None and "mesh" in outputs:
        generated_mesh = outputs["mesh"][0] if isinstance(outputs["mesh"], list) else outputs["mesh"]
        save_mesh(generated_mesh, high_mesh_path)
        trimesh.load(high_mesh_path).export(mesh_path)
        return
    if mesh_source is None:
        raise RuntimeError("SAM 3D did not return a mesh/glb output; cannot continue pose estimation.")

    if isinstance(mesh_source, trimesh.Scene):
        mesh_source.export(mesh_path, file_type="obj")
        mesh_source.export(high_mesh_path, file_type="obj")
    else:
        mesh_source.export(mesh_path)
        mesh_source.export(high_mesh_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM 3D Objects generation in its own environment.")
    parser.add_argument("--sam3d-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--mesh-path", required=True)
    parser.add_argument("--high-mesh-path", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--splat-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    Inference, ready_gaussian_for_video_rendering, render_sam3d_video = load_sam3d_notebook_api(args.sam3d_root)

    config_path = os.path.join(args.sam3d_root, "checkpoints", "hf", "pipeline.yaml")
    inference = Inference(config_path, compile=False)

    image_np = np.array(Image.open(args.image).convert("RGB")).astype(np.uint8)
    mask_np = np.array(Image.open(args.mask).convert("L")) > 0
    outputs = inference(image_np, mask_np, seed=args.seed)

    if "gs" in outputs:
        outputs["gs"].save_ply(args.splat_path)
        scene_gs = ready_gaussian_for_video_rendering(outputs["gs"], in_place=False, fix_alignment=True)
        video_geo = render_sam3d_video(scene_gs, resolution=1024, num_frames=120)
        imageio.mimsave(args.video_path, video_geo, fps=15)

    export_sam3d_mesh(outputs, args.mesh_path, args.high_mesh_path)


if __name__ == "__main__":
    main()
