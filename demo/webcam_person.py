# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import warnings
import time
import copy
import circular

import cv2
import mmcv
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pyskl.apis import inference_recognizer, init_recognizer, inference_recognizer_person

try:
    from mmdet.apis import inference_detector, init_detector #
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )
from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='Webcam demo')
    parser.add_argument(
        '--config',
        default='../configs/stgcn++/stgcn++_ntu30_xsub_hrnet/j.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default=('../work_dirs/stgcn++/stgcn++_new_ntu30_xsub_hrnet/j/epoch_16.pth'),
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    args = parser.parse_args()
    return args

def detection_inference(args, model, frame_paths):

    assert model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'Note that you should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
    result = inference_detector(model, frame_paths)
    # We only keep human detections with score larger than det_score_thr
    result = result[0][result[0][:, 4] >= args.det_score_thr]
    return result


def pose_inference(args, model, frame_paths, det_results):

    # Align input format
    det_results = [dict(bbox=x) for x in list(det_results)]
    pose = inference_top_down_pose_model(model, frame_paths, det_results, format='xyxy')[0]
    return pose


def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]


def main():
    args = parse_args()

    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    # Are we using GCN for Infernece?
    GCN_flag = False #'GCN' in config.model.type
    GCN_nperson = None
    #if GCN_flag:
    #    format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
    #    GCN_nperson = format_op['num_person']

    # model load
    model = init_recognizer(config, args.checkpoint, args.device)
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    det_model = init_detector(args.det_config, args.det_checkpoint, args.device)

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    video_capture = cv2.VideoCapture(0)
    mirror = 1
    num_frame = 30

    # Get Human detection results
    det_results = circular.CircularQueue(num_frame)
    pose_results = circular.CircularQueue(num_frame)

    while(True):
        #get frame
        ret, frame_paths = video_capture.read()
        h, w, _ = frame_paths.shape

        # Get clip_len, frame_interval and calculate center index of each clip
        if GCN_flag:
            format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
            GCN_nperson = format_op['num_person']

        det_result = detection_inference(args,det_model, frame_paths)
        torch.cuda.empty_cache()

        pose_result = pose_inference(args, pose_model,frame_paths, det_result)
        torch.cuda.empty_cache()

        det_results.enqueue(copy.copy(det_result))
        pose_results.enqueue(copy.copy(pose_result))

        if det_results.size() >= num_frame:
            fake_anno = dict(
                frame_dir='',
                label=-1,
                img_shape=(h, w),
                original_shape=(h, w),
                start_index=0,
                modality='Pose',
                total_frames=num_frame)

            tracking_inputs = []
            if GCN_flag:
                # We will keep at most `GCN_nperson` persons per frame.

                print(pose_results)
                for poses in pose_results:
                    print(poses)
                    for pose in poses:
                        if len(pose[0])>0:

                            a = dict(pose[0])

                            tracking_inputs+=[a['keypoints']]
                #tracking_inputs = [[pose['keypoints'] for pose in poses] for poses in pose_results]

                keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=GCN_nperson)
                fake_anno['keypoint'] = keypoint
                fake_anno['keypoint_score'] = keypoint_score
            else:
                num_person = pose_results.max_len()
                num_keypoint = 17
                keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                                    dtype=np.float16)
                keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                          dtype=np.float16)
                for i in range(pose_results.size()):
                    poses = pose_results.value(i)
                    for j, pose in enumerate(poses):
                        pose = pose['keypoints']
                        keypoint[j, i] = pose[:, :2]
                        keypoint_score[j, i] = pose[:, 2]
                fake_anno['keypoint'] = keypoint
                fake_anno['keypoint_score'] = keypoint_score

            output = inference_recognizer_person(model, fake_anno)

            h = 50
            vis_frames = vis_pose_result(pose_model, frame_paths, pose_result)
            for i, p_label in enumerate(output):
                voting_label_name = label_map[p_label]
                h += 20
                cv2.putText(vis_frames, 'person' + str(i) + ': ' + voting_label_name, (10, h), FONTFACE, FONTSCALE, FONTCOLOR,
                            THICKNESS, LINETYPE)

            cv2.namedWindow("ST-GCN++", flags=cv2.WINDOW_NORMAL)
            cv2.imshow("ST-GCN++", vis_frames)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            det_results.dequeue()
            pose_results.dequeue()

if __name__ == '__main__':
    main()
