import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def sample_and_group_all_gt(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = xyz
    grouped_xyz = xyz.view(B, 1, N, C)

    relative_coord = grouped_xyz[:, :, None, :, :] - grouped_xyz[:,:, :, None, :]

    pos_dist_sqrt=relative_coord

    if points is not None:

        new_points =points.view(B, 1, N, -1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, pos_dist_sqrt

class PatchMerging_avepooling(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.avepooling_layer=nn.AdaptiveAvgPool2d((n,1))
        ####################################################

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)
        x = self.avepooling_layer(x)
        x = x.permute(0, 2, 3, 1)
        
        return x


class WindowAttention(nn.Module):
    def __init__(self, input_dim, output_dim, heads, head_dim, patch_size, attn_drop_value):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.patch_size = patch_size
        self.input_dim=input_dim 
        self.output_dim=output_dim

        self.linear_input = nn.Sequential(nn.Linear(input_dim, output_dim), nn.BatchNorm2d(output_dim), nn.ReLU(inplace=True))
        self.linear_out = nn.Sequential(nn.BatchNorm2d(output_dim), nn.ReLU(inplace=True))

        self.q_conv = nn.Conv2d(input_dim, output_dim, 1)
        self.k_conv = nn.Conv2d(input_dim, output_dim, 1)


        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm3d(3), nn.ReLU(inplace=True), nn.Linear(3, head_dim))
        self.linear_dots = nn.Sequential(nn.Linear(head_dim, head_dim), nn.ReLU(inplace=True), nn.Linear(head_dim, head_dim))


        self.v_conv = nn.Conv2d(input_dim, output_dim, 1)


        self.trans_conv = nn.Conv2d(output_dim, output_dim, 1)
        self.after_norm = nn.BatchNorm2d(output_dim)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, p_r):


        shortcut=x

        b, n, K, _, h = *x.shape, self.heads

        for i, layer in enumerate(self.linear_input): x = layer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) if i == 1 else layer(x)

        
        x_q = self.q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        x_k = self.k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_v = self.v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # b, n, k, c


        x_q = rearrange(x_q, 'b n k (h d) -> b h n k d',
                        h=h, k=K)
        x_k = rearrange(x_k, 'b n k (h d) -> b h n k d',
                        h=h, k=K)
        x_v = rearrange(x_v, 'b n k (h d) -> b h n k d',
                        h=h, k=K)

        # b,h,n,k,d

        dots = x_q[:,:,:, :, None, :]-x_k[:, :,:, None, :, :]
        # b,h,n,k,k

        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1) if i == 1 else layer(p_r)
        # b,n,k,k,1

        p_r=p_r.unsqueeze(-1)
        # b,n,k,k, d,1

        p_r=p_r.permute(0,5,1,2,3,4)

        # b,1,n,k,k
        dots=dots+p_r

        for i, layer in enumerate(self.linear_dots): dots =  layer(dots)


        attention = self.softmax(dots)
        attention = attention / (1e-9 + attention.sum(dim=-2, keepdim=True)) 
        # b,h,n,k,k

        x_v=x_v.unsqueeze(-1)
        #b,h,n,k,d,1
        x_v=x_v.permute(0,1,2,5,3,4)
        #b,h,n,1,k,d

        out = attention * x_v
        # b,h,n,k,k,d
        # b,h,n,k,d

        out = rearrange(out, 'b h n k1 k2 d -> b n k1 k2 (h d)',
                                h=h, k1=K)
        # b,n,k,k,c
        out = torch.sum(out, dim=-2, keepdim=False)
        # b,n,k,c

        for i, layer in enumerate(self.linear_out): out = layer(out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) if i == 0 else layer(out)




        x_r = shortcut.permute(0, 3, 1, 2) + self.act(self.after_norm(self.trans_conv(shortcut.permute(0, 3, 1, 2) - out.permute(0, 3, 1, 2))))


        x_r=x_r.permute(0, 2, 3, 1)



       

        return x_r

class SwinBlock(nn.Module):
    def __init__(self, batchnorm, input_dim, output_dim, heads, head_dim, mlp_dim, patch_size, attn_drop_value, feed_drop_value):
        super().__init__()
        self.attention_block = WindowAttention(input_dim=input_dim, 
                                                output_dim=output_dim,
                                                heads=heads,
                                                head_dim=head_dim,
                                                patch_size=patch_size,
                                                attn_drop_value=attn_drop_value)

    def forward(self, x, pos_dist_sqrt):
        x = self.attention_block(x, pos_dist_sqrt)
        return x

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)), inplace=True)

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class Global_Transformer(nn.Module):
    def __init__(self, avepooling, batchnorm, attn_drop_value, feed_drop_value, npoint, in_channel, out_channels, layers, num_heads, head_dim):
        super(Global_Transformer, self).__init__()
        self.npoint = npoint
        self.avepooling = avepooling

        

        
        self.layer = SwinBlock(batchnorm, input_dim=in_channel, output_dim=out_channels, heads=num_heads, head_dim=head_dim, mlp_dim=in_channel * 4,
                          patch_size=npoint, attn_drop_value=attn_drop_value, feed_drop_value=feed_drop_value)
        
        self.avepooling_layer=PatchMerging_avepooling(1)
        #################################################################################################

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        

        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        new_xyz, new_points, pos_dist_sqrt = sample_and_group_all_gt(xyz, points)
        # new_xyz: sampled points position data, [B, npoint, 3]
        # new_points: sampled points data, [B, 1, npoint, D]
        B, S, _ = new_xyz.shape
        ######################################################################
        
 
        new_points=self.layer(new_points, pos_dist_sqrt)
        if self.avepooling:
            new_points=self.avepooling_layer(new_points)
            new_points=new_points.view(B,1,-1) # B,1, D

        else:
            new_points=new_points.contiguous().view(B,S,-1) # B,S, D
        #####################################################################
        
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, dim1,in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = mlp[0]
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.res1 = nn.Conv1d(in_channel,mlp[0] , 1, 1)
        self.batchn = nn.BatchNorm1d(mlp[0])

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = F.relu(self.batchn(self.res1(new_points)))
        res = new_points
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = bn(conv(new_points))

        return F.relu(new_points + res)
