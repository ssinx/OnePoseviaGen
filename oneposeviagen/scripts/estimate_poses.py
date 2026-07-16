from fpose.estimater import *


def estimate_poses(npz_path, query_image_names, query_depth_names, query_mask_names, query_intrinsics, scaled_model_path, output_dir, debug=0, est_refine_iter=5, anchor_index=0):
    #estimate the poses of the query images
    debug_dir = output_dir
    mesh = trimesh.load(scaled_model_path, force='mesh')
    frame_count = len(query_image_names)
    if frame_count == 0:
        raise ValueError("No query images were provided")
    if not 0 <= anchor_index < frame_count:
        raise ValueError(f"anchor_index {anchor_index} is out of range for {frame_count} frames")
    
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    poses = [None] * frame_count
    rgbs = [None] * frame_count
    depths = [None] * frame_count

    def load_frame(frame_id):
        color = cv2.imread(query_image_names[frame_id])
        # depth = reader.get_depth(i)
        depth = cv2.imread(query_depth_names[frame_id],-1)/1e3
        # mask = reader.get_mask(0).astype(bool)
        mask = cv2.imread(query_mask_names[frame_id],-1)
        K = np.array(query_intrinsics[frame_id])
        if len(mask.shape)==3:
            for c in range(3):
                if mask[...,c].sum()>0:
                    mask = mask[...,c]
                    break
        mask = mask.astype(bool)
        return color, depth, mask, K

    color, depth, mask, K = load_frame(anchor_index)
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
    poses[anchor_index] = pose.reshape(4, 4)
    if est.pose_last is None:
        raise RuntimeError(f"Failed to initialize pose at anchor_index={anchor_index}; check anchor mask/depth validity")
    anchor_pose_last = est.pose_last.clone()

    for frame_id in range(anchor_index + 1, frame_count):
        color, depth, _, K = load_frame(frame_id)
        pose = est.track_one(K=K, rgb=color, depth=depth, iteration=est_refine_iter, extra={})
        poses[frame_id] = pose.reshape(4, 4)

    est.pose_last = anchor_pose_last.clone()
    for frame_id in range(anchor_index - 1, -1, -1):
        color, depth, _, K = load_frame(frame_id)
        pose = est.track_one(K=K, rgb=color, depth=depth, iteration=est_refine_iter, extra={})
        poses[frame_id] = pose.reshape(4, 4)

    for frame_id, pose in enumerate(poses):
        color, depth, _, K = load_frame(frame_id)
        center_pose = pose@np.linalg.inv(to_origin)
        rgb_vis, dep_vis = draw_posed_3d_box_with_depth(K, img=color, depth=depth, ob_in_cam=center_pose, bbox=bbox)
        rgb_new, dep_new = draw_xyz_axis_with_depth(rgb_vis, depth=dep_vis, ob_in_cam=center_pose, scale=0.3, K=K, thickness=3, transparency=0, is_input_rgb=True)
        rgb_new = np.transpose(rgb_new, (2, 0, 1))/255
        rgbs[frame_id] = rgb_new
        depths[frame_id] = dep_new

    data = dict(np.load(npz_path))
    # 替换指定字段
    data['depths'] = depths
    data['video'] = rgbs
    np.savez(npz_path, **data)
    return poses  # Return the list of estimated poses for each image
