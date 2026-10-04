r"""
    Preprocess DIP-IMU and TotalCapture test dataset.
    Synthesize AMASS dataset.

    Please refer to the `paths` in `config.py` and set the path of each dataset correctly.
"""

# BEGIN D-BRAN PROJECT BOOTSTRAP
import sys as _dbran_sys
from pathlib import Path as _DbranPath

_dbran_current_file = _DbranPath(__file__).resolve()

for _dbran_candidate in (
    _dbran_current_file.parent,
    *_dbran_current_file.parents,
):
    if (_dbran_candidate / "main_path.py").is_file():
        _dbran_root_string = str(_dbran_candidate)

        if _dbran_root_string not in _dbran_sys.path:
            _dbran_sys.path.insert(0, _dbran_root_string)

        break
else:
    raise RuntimeError(
        "Could not locate the D-BRAN project root from "
        f"{_dbran_current_file}"
    )

from main_path import PROJECT_ROOT
# END D-BRAN PROJECT BOOTSTRAP


import articulate as art
import torch
import os
import pickle
from config import paths, amass_data
import numpy as np
from tqdm import tqdm
import glob

# Global configuration for single sequence export
k = 1  # Sequence index to export for Unity visualization

vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
body_model = art.ParametricModel(paths.smpl_file)


def _syn_acc(v, smooth_n=4):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def process_amass():
    data_pose, data_trans, data_beta, length = [], [], [], []
    raw_seq_names = []  # To track all loaded names

    for ds_name in amass_data:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try:
                cdata = np.load(npz_fname)
            except:
                continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120:
                step = 2
            elif framerate == 60 or framerate == 59:
                step = 1
            else:
                continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])
            raw_seq_names.append(os.path.basename(npz_fname))

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    
    kept_seq_names = [] # Names of sequences that are NOT discarded

    for i, l in tqdm(list(enumerate(length))):
        if l <= 12:
            b += l
            print('\tdiscard one sequence with length', l)
            continue
        
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())      # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
        
        kept_seq_names.append(raw_seq_names[i]) # Keep track of valid sequence name
        b += l

    print('Saving')
    os.makedirs(paths.amass_dir, exist_ok=True)
    torch.save(out_pose, os.path.join(paths.amass_dir, 'pose.pt'))
    torch.save(out_shape, os.path.join(paths.amass_dir, 'shape.pt'))
    torch.save(out_tran, os.path.join(paths.amass_dir, 'tran.pt'))
    torch.save(out_joint, os.path.join(paths.amass_dir, 'joint.pt'))
    torch.save(out_vrot, os.path.join(paths.amass_dir, 'vrot.pt'))
    torch.save(out_vacc, os.path.join(paths.amass_dir, 'vacc.pt'))
    print('Synthetic AMASS dataset is saved at', paths.amass_dir)

    # Export specific sequence for Unity visualization
    if len(out_vacc) > 0:
        if not (0 <= k < len(out_vacc)):
            print(f"[warning] k = {k} is out of range (0..{len(out_vacc)-1}). Skipping single-sequence export.")
        else:
            print(f"[info] AMASS: Selected k = {k}, Sequence = {kept_seq_names[k]}")
            print(f"[info] Length: {out_vacc[k].shape[0]} frames")

            acc_one = out_vacc[k].contiguous()  # (T,6,3)
            ori_one = out_vrot[k].contiguous()  # (T,6,3,3)
            torch.save(acc_one, os.path.join(paths.amass_dir, 'vacc_AMASS.pt'))
            torch.save(ori_one, os.path.join(paths.amass_dir, 'vrot_AMASS.pt'))
            print(f"[info] AMASS: ACC and ORI for Unity (single sequence) saved.")
    else:
        print("[warning] No AMASS sequences kept; cannot export single-sequence files.")


def process_dipimu():
    imu_mask = [7, 8, 11, 12, 0, 2]

    train_split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
    test_split  = ['s_09', 's_10']

    def _process_subjects(subject_list):
        accs, oris, poses, trans = [], [], [], []
        all_filenames = []

        for subject_name in subject_list:
            subj_dir = os.path.join(paths.raw_dipimu_dir, subject_name)
            if not os.path.isdir(subj_dir):
                print(f"DIP-IMU: missing subject dir: {subj_dir} (skip)")
                continue

            for motion_name in os.listdir(subj_dir):
                path = os.path.join(subj_dir, motion_name)
                data = pickle.load(open(path, 'rb'), encoding='latin1')

                acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
                ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
                pose = torch.from_numpy(data['gt']).float()

                # Fill nan with nearest neighbors
                for _ in range(4):
                    acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                    ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                    acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                    ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

                # Remove 6 frames at start/end (same as original)
                acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]

                if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                    accs.append(acc.clone())
                    oris.append(ori.clone())
                    poses.append(pose.clone())
                    trans.append(torch.zeros(pose.shape[0], 3))  # DIP-IMU has no translations
                    all_filenames.append(f"{subject_name}/{motion_name}")
                else:
                    print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

        return accs, oris, poses, trans, all_filenames

    os.makedirs(paths.dipimu_dir, exist_ok=True)

    # Build train.pt
    tr_accs, tr_oris, tr_poses, tr_trans, tr_names = _process_subjects(train_split)
    torch.save(
        {'acc': tr_accs, 'ori': tr_oris, 'pose': tr_poses, 'tran': tr_trans},
        os.path.join(paths.dipimu_dir, 'train.pt')
    )
    print('Preprocessed DIP-IMU TRAIN saved at', os.path.join(paths.dipimu_dir, 'train.pt'))
    print('DIP-IMU TRAIN sequences:', len(tr_accs))

    # Build test.pt
    te_accs, te_oris, te_poses, te_trans, te_names = _process_subjects(test_split)
    torch.save(
        {'acc': te_accs, 'ori': te_oris, 'pose': te_poses, 'tran': te_trans},
        os.path.join(paths.dipimu_dir, 'test.pt')
    )
    print('Preprocessed DIP-IMU TEST saved at', os.path.join(paths.dipimu_dir, 'test.pt'))
    print('DIP-IMU TEST sequences:', len(te_accs))

    # Optional: Export one sequence for Unity (from TEST by default)
    # Define k before calling process_dipimu(), or set k here.
    try:
        k  # noqa: F821
        if len(te_accs) > 0:
            if not (0 <= k < len(te_accs)):
                print(f"[warning] k = {k} is out of range (0..{len(te_accs)-1}). Skipping single-sequence export.")
            else:
                print(f"[info] DIP-IMU TEST: Selected k = {k}, File = {te_names[k]}")
                acc_one = te_accs[k].contiguous()  # (T,6,3)
                ori_one = te_oris[k].contiguous()  # (T,6,3,3)
                torch.save(acc_one, os.path.join(paths.dipimu_dir, 'acc_DIP.pt'))
                torch.save(ori_one, os.path.join(paths.dipimu_dir, 'ori_DIP.pt'))
                print("[info] DIP-IMU: ACC and ORI for Unity (single sequence) saved.")
        else:
            print("[warning] No DIP-IMU test sequences available; cannot export single-sequence files.")
    except NameError:
        pass


def process_totalcapture():
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'

    accs, oris, poses, trans = [], [], [], []
    files_dip = []  # Map sequence to DIP file name

    for file in sorted(os.listdir(paths.raw_totalcapture_dip_dir)):
        data = pickle.load(open(os.path.join(paths.raw_totalcapture_dip_dir, file), 'rb'), encoding='latin1')
        
        # [FIX] permutation removed here.
        ori = torch.from_numpy(data['ori']).float()
        acc = torch.from_numpy(data['acc']).float()
        
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3
        files_dip.append(file) # Track filename

    for subject_name in ['S1', 'S2', 'S3', 'S4', 'S5']:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official_dir, subject_name))):
            if subject_name == 'S5' and motion_name == 'acting3':
                continue   # no SMPL poses
            f = open(os.path.join(paths.raw_totalcapture_official_dir, subject_name, motion_name, file_name))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])   # N, 3

    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    # remove acceleration bias
    for iacc, pose, tran in zip(accs, poses, trans):
        pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        _, _, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        vacc = _syn_acc(vert[:, vi_mask])
        for imu_id in range(6):
            for i in range(3):
                d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
                iacc[:, imu_id, i] += d

    os.makedirs(paths.totalcapture_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans},
               os.path.join(paths.totalcapture_dir, 'test.pt'))
    print('Preprocessed TotalCapture dataset is saved at', paths.totalcapture_dir)

    # Export specific sequence for Unity visualization
    if len(accs) > 0:
        if not (0 <= k < len(accs)):
            print(f"[warning] k = {k} is out of range (0..{len(accs)-1}). Skipping single-sequence export.")
        else:
            print(f"[info] TotalCapture: Selected k = {k}, File = {files_dip[k]}")

            acc_one = accs[k].contiguous()  # (T,6,3)
            ori_one = oris[k].contiguous()  # (T,6,3,3)
            torch.save(acc_one, os.path.join(paths.totalcapture_dir, 'acc_TotalCapture.pt'))
            torch.save(ori_one, os.path.join(paths.totalcapture_dir, 'ori_TotalCapture.pt'))
            print("[info] TotalCapture: ACC and ORI for Unity (single sequence) saved.")
    else:
        print("[warning] No TotalCapture sequences available; cannot export single-sequence files.")


if __name__ == '__main__':
    # Uncomment the function you want to run
    process_amass()
    # process_dipimu()
    # process_totalcapture()