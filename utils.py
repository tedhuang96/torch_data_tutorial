import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm.notebook import tqdm
import time


def anorm(p1, p2):
    NORM = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    if NORM == 0:
        return 0
    return 1/(NORM)


def seq_to_graph(seq_, seq_rel, attn_mech='glob_kip'):
    """
    inputs:
        - seq_: global positions. tensor: (1, num_peds, 2, seq_len)
        - seq_rel: displacements. tensor: (1, num_peds, 2, seq_len)
        - attn_mech # attention mechanism
        # 'glob_kip' means kip normalization.
        # 'auth' means the original code from the stgcnn author.
    """
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0] # num_peds

    V = np.zeros((seq_len, max_nodes, 2))

    for s in range(seq_len):
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_rel)):
            V[s, h, :] = step_rel[h] # (2,)

    if attn_mech == 'glob_kip':
        A = np.zeros((seq_len, max_nodes, max_nodes))
        for s in range(seq_len):
            step_ = seq_[:, :, s]
            for h in range(len(step_)):
                for k in range(h+1, len(step_)):
                    # bug_fix # global instead of relative positions
                    l2_norm = anorm(step_[h], step_[k])
                    A[s, h, k] = l2_norm
                    A[s, k, h] = l2_norm
            A_hat = A[s, :, :]+np.eye(max_nodes)
            D_hat = np.eye(max_nodes)*max_nodes  # fully connected graph
            D_half_inv = np.linalg.inv(np.sqrt(D_hat))
            # bug_fix # Kip_normalization
            Kip_normalization = np.matmul(np.matmul(D_half_inv, A_hat), D_half_inv)
            A[s, :, :] = Kip_normalization
    elif attn_mech == 'auth':
        A = np.zeros((seq_len, max_nodes, max_nodes))
        for s in range(seq_len):
            step_ = seq_[:, :, s]
            for h in range(len(step_)):
                for k in range(h+1, len(step_)):
                    l2_norm = anorm(step_[h], step_[k])
                    A[s, h, k] = l2_norm
                    A[s, k, h] = l2_norm
            A_hat = A[s, :, :]+np.eye(max_nodes)
            # G = nx.from_numpy_matrix(A[s, :, :])
            G = nx.DiGraph(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()
            # zhe # Someone on the github issues and I spotted that they actually use the function wrong.
            # zhe # so they actually have a negative value for the attention.
    else:
        print('Wrong attention mechanism.')
        sys.exit(1)

    return torch.from_numpy(V).type(torch.float),\
        torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_ped=1, delim='\t', attn_mech='glob_kip'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files

        - attn_mech # attention mechanism
        # 'glob_kip' means kip normalization.
        # 'auth' means the original code from the stgcnn author.
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.attn_mech = attn_mech

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        print(all_files)
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(
                    self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros(
                    (len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros(
                    (len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(
            non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]
            v_, a_ = seq_to_graph(
                self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], self.attn_mech)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_, a_ = seq_to_graph(
                self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], self.attn_mech)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]
        ]
        return out


def create_datasets(args, pkg_path, save_datasets=True):
    """
    Create train, val and test datasets.
    inputs:
        - args
            - obs_seq_len: 8
            - pred_seq_len: 12
            - dataset: eth, hotel, univ, zara1, zara2
            - attn_mech: attention mechanism. e.g. glob_kip, plain.
        - pkg_path: package path
        - save_datasets: bool.
    outputs:
        - dsets
            - ['train']: training dataset
            - ['val']: validation dataset
            - ['test']: test dataset
    """
    obs_seq_len = args.obs_seq_len
    pred_seq_len = args.pred_seq_len
    dataset_folderpath = os.path.join(pkg_path, 'datasets', args.dataset)
    subfolders = ['train', 'val', 'test']
    dsets = {}
    for subfolder in subfolders:
        dset = TrajectoryDataset(
            os.path.join(dataset_folderpath, subfolder),
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            attn_mech=args.attn_mech)
        if save_datasets:
            result_filename = args.dataset+'_dset_'+subfolder+'_'+args.attn_mech+'.pt'
            torch.save(dset, os.path.join(dataset_folderpath, result_filename))
            print(os.path.join(dataset_folderpath, result_filename)+' is created.')
        dsets[subfolder] = dset
    return dsets

def create_datasets_test(args, pkg_path, save_datasets=True):
    """
    Create test datasets only. For tutorial.
    inputs:
        - args
            - obs_seq_len: 8
            - pred_seq_len: 12
            - dataset: eth, hotel, univ, zara1, zara2
            - attn_mech: attention mechanism. e.g. glob_kip, auth.
        - pkg_path: package path
        - save_datasets: bool.
    outputs:
        - dsets
            - ['test']: test dataset
    """
    obs_seq_len = args.obs_seq_len
    pred_seq_len = args.pred_seq_len
    dataset_folderpath = os.path.join(pkg_path, 'datasets', args.dataset)
    subfolders = ['test']
    dsets = {}
    for subfolder in subfolders:
        dset = TrajectoryDataset(
            os.path.join(dataset_folderpath, subfolder),
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            attn_mech=args.attn_mech)
        if save_datasets:
            result_filename = args.dataset+'_dset_'+subfolder+'_'+args.attn_mech+'.pt'
            torch.save(dset, os.path.join(dataset_folderpath, result_filename))
            print(os.path.join(dataset_folderpath, result_filename)+' is created.')
        dsets[subfolder] = dset
    return dsets

def load_dataset(args, pkg_path, subfolder='train', num_workers=1):
    """
    load datasets into a DataLoader object.
    inputs:
        - args
            - dataset: eth, hotel, univ, zara1, zara2
            - attn_mech: attention mechanism. e.g. glob_kip, plain.
        - pkg_path: package path
        - subfolder: 'train', 'val', 'test'
        - num_workers: 0, 1, or 4 may be the best. Need test.
    outputs:
        - dloader
    """
    result_filename = args.dataset+'_dset_'+subfolder+'_'+args.attn_mech+'.pt'
    dataset_folderpath = os.path.join(pkg_path, 'datasets', args.dataset)
    dset = torch.load(os.path.join(dataset_folderpath, result_filename))
    print(os.path.join(dataset_folderpath, result_filename)+' is loaded.')
    if subfolder == 'train':
        shuffle = True
    else:
        shuffle = False
    dloader = DataLoader(
        dset,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=shuffle,
        num_workers=num_workers)
    return dloader


def dataset_format():
    """
    Documentation on structure of a batch from the dataset.
    - batch # list.

        - obs_traj # global positions in observation. # tensor: (1, num_peds, 2, 8)
        # 2 means x and y. # 8 means obs_period.

        - pred_traj_gt # ground truth global positions in prediction. # tensor: (1, num_peds, 2, 12)
        # 12 means pred_period.

        - obs_traj_rel # displacement in observation. # tensor: (1, num_peds, 2, 8)
        # obs_traj_rel[:,:,:,0] is zero. # obs_traj_rel[:,:,:,1:]=obs_traj[:,:,:,1:]-obs_traj[:,:,:,:-1]

        - pred_traj_gt_rel # ground truth displacement in prediction. # tensor: (1, num_peds, 2, 12)
        # pred_traj_gt_rel[:,:,:,0]=pred_traj_gt[:,:,:,0]-obs_traj[:,:,:,-1]

        - non_linear_ped # 0-1 vector indicates whether motion is nonlinear. # tensor: (1, num_peds)

        - loss_mask # all-one tensor. # tensor: (1, num_peds, 20) # 20 means full_period.

        - V_obs # vertices represent displacement of pedestrians in observation.
        # tensor: (1, 8, num_peds, 2) # V_obs.permute(0,2,3,1) == obs_traj_rel

        - A_obs # Adjacency matrix of pedestrians in observation. # tensor: (1, 8, num_peds, num_peds)

        - V_tr # ground truth vertices represent displacement of pedestrians in prediction.
        # tensor: (1, 12, num_peds, 2)

        - A_tr # ground truth Adjacency matrix of pedestrians in prediction.
        # tensor: (1, 12, num_peds, num_peds)
    """
    pass
