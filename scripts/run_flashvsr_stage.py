import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run FlashVSR on an image sequence and export one enhanced frame.")
    parser.add_argument("--flashvsr-script", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-frame-index", type=int, required=True)
    return parser.parse_args()


def load_flashvsr_module(script_path):
    script_path = Path(script_path).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"FlashVSR script not found: {script_path}")

    flashvsr_root = str(script_path.parent)
    if flashvsr_root not in sys.path:
        sys.path.insert(0, flashvsr_root)
    os.chdir(flashvsr_root)

    spec = importlib.util.spec_from_file_location("flashvsr_inference", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input frame directory not found: {input_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flashvsr = load_flashvsr_module(args.flashvsr_script)
    lq_video, height, width, frame_count, _ = flashvsr.prepare_input_tensor(
        str(input_dir),
        scale=4.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    pipeline = flashvsr.init_pipeline()
    try:
        output_video = pipeline(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=0,
            LQ_video=lq_video,
            num_frames=frame_count,
            height=height,
            width=width,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=2.0 * 768 * 1280 / (height * width),
            kv_ratio=3.0,
            local_range=11,
            color_fix=True,
        )
        output_frames = flashvsr.tensor2video(output_video)
        if not 0 <= args.target_frame_index < len(output_frames):
            raise RuntimeError(
                f"Requested frame {args.target_frame_index}, but FlashVSR returned {len(output_frames)} frames."
            )
        output_frames[args.target_frame_index].save(output_path)
    finally:
        del pipeline
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print(f"Saved FlashVSR frame {args.target_frame_index}: {output_path}")


if __name__ == "__main__":
    main()
