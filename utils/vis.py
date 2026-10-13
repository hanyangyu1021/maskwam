import copy
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
import seaborn as sns
import numpy as np
import math
import re
import cv2
import torch
from io import BytesIO
from PIL import Image
from scipy.spatial.transform import Rotation
# pip install adjustText
from adjustText import adjust_text

def visualize_attention_mask(attention_mask, batch_index=0, head_index=0):
    '''
    :param attention_mask: (bs, heads, q_len, k_len)
    :param batch_index:
    :param head_index:
    :return:
    '''
    if len(attention_mask.shape) == 3:
        mask_to_visualize = attention_mask[batch_index]  # 形状变为 (query_length, key_value_length)
    elif len(attention_mask.shape) == 4:
        mask_to_visualize = attention_mask[batch_index, head_index]  # 形状变为 (query_length, key_value_length)
    # print(mask_to_visualize)

    plt.figure(figsize=(32, 32))
    sns.heatmap(mask_to_visualize.float().cpu().detach().numpy(), cmap='viridis', cbar=True, linecolor='black', square=False)
    plt.title(f'Attention Mask - Batch {batch_index}, Head {head_index}')
    plt.xlabel('Key/Value Length')
    plt.ylabel('Query Length')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()  # 将 x 轴刻度放到顶部
    # plt.grid()
    # plt.gca().invert_xaxis() # as kv is format as [kv cache] + [current kv]
    plt.show()
    plt.savefig(f'attention_mask.png')
    plt.close()

def get_parameters(model, name):
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(f"{name}: Total parameters: {total_params} M")
    print(f"{name}: Trainable parameters: {trainable_params} M")

def get_shape(t):
    if isinstance(t, tuple):
        return (len(t),) + get_shape(t[0]) if t else (0,)
    elif isinstance(t, list):
        return (len(t),) + get_shape(t[0]) if t else (0,)
    else:
        return ()

def save_point_cloud_to_obj(point_cloud, filename):
    with open(filename, 'w') as f:
        for point in point_cloud:
            # 写入每个点的坐标
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")

#from lerobot
def format_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num

# def vis_atten_map(att_output, images, wandb=False):
#     # images [[b,c,h,w],[b,c,h,w],[b,c,h,w]]
#     # att_output [[b, num_head, q_len, k_len]]
#     ########################### check attention #############################
#
#     images_vis = images[:]
#     for i, img in enumerate(images_vis):
#         # (-1,1) -> (0, 1)
#         images_vis[i] = (images_vis[i]*0.5)+0.5
#         # chw 2 hwc
#         images_vis[i] = images_vis[i].permute(0,2,3,1).float().detach().cpu().numpy()
#
#     def overlay(rgb, heatmap):
#         heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_NEAREST)
#         heatmap = plt.cm.jet(heatmap)[:, :, :3]  # 只取 RGB 通道
#         heatmap = (heatmap * 255).astype(np.uint8)  # 转换为 uint8 类型
#         rgb = cv2.resize(rgb, (224, 224))
#         if np.max(rgb)<2:
#             rgb *= 255.
#         rgb = rgb.astype(np.uint8)
#
#         overlay = cv2.addWeighted(rgb, 0.5, heatmap, 0.5, 0)
#         return overlay
#
#     bs = 0
#     num_views = len(images_vis)
#     num_layers = len(att_output)
#     num_heads = att_output[bs].shape[1]
#     patch_size = 16
#     num_patches = patch_size * patch_size
#     for layer in range(1, num_layers):
#         # map = output_action["attentions"][layer].view(1, 1, 51 * 8, -1)
#         # plt.imshow(map[0, 0, :, :].float().cpu().numpy(), cmap='viridis')
#         # for i in range(8):
#         #     plt.subplot(8, 1, i + 1)
#         #     map = output_action["attentions"][layer][0, i, :, :].float().cpu().numpy()
#         #     plt.imshow(map, cmap='viridis')
#         # plt.show()
#
#         fig, axes = plt.subplots(num_views, num_heads + 1, figsize=(4 * (num_heads + 1), 4 * 3))
#         for view in range(num_views):
#             for i in range(num_heads + 1):
#                 ax = axes[view, i]
#                 if i == num_heads:
#                     map = images_vis[view][bs]
#                 else:
#                     map = att_output[layer][bs, i, 1, :num_patches].reshape(patch_size,
#                                                                             patch_size).float().detach().cpu().numpy()
#                     map = (map - np.min(map)) / (np.max(map) - np.min(map))
#                     map = overlay(images_vis[view][bs], map)
#                 ax.imshow(map)
#                 ax.set_title(f'View{view + 1} Head Attention{i + 1}' if i < num_heads else f'View{view + 1} ')
#                 ax.axis('off')
#
#         plt.show()
#         if wandb:
#             return fig
#         # 保存图片，建议文件命名带上layer序号
#         plt.tight_layout()
#         save_dir = "./attention_maps"
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(f'{save_dir}/attention_layer_{layer}.png', dpi=200)  # 可改文件夹，如'./att_plot/attention_layer_{layer+19}.png'
#         plt.close()  # 不需要show，否则弹出窗口；close可以释放内存


def vis_atten_map(att_output, images, wandb=False):
    def overlay(rgb, heatmap):
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_NEAREST)
        heatmap = plt.cm.jet(heatmap)[:, :, :3]  # 只取 RGB 通道
        heatmap = (heatmap * 255).astype(np.uint8)  # 转换为 uint8 类型
        rgb = cv2.resize(rgb, (224, 224))
        if np.max(rgb)<2:
            rgb = rgb.astype(np.float32) * 255.
        rgb = rgb.astype(np.uint8)

        overlay = cv2.addWeighted(rgb, 0.5, heatmap, 0.5, 0)
        return overlay

    def concat_layers_imgs_auto(layer_figs, n_cols=6):  # 默认每行6个
        layer_num = len(layer_figs)
        n_rows = math.ceil(layer_num / n_cols)

        # 假设所有图片大小一致
        img_width, img_height = layer_figs[0].size

        composite = Image.new('RGB', (n_cols * img_width, n_rows * img_height),(255, 255, 255))

        for idx, im in enumerate(layer_figs):
            row = idx // n_cols
            col = idx % n_cols
            x = col * img_width
            y = row * img_height
            composite.paste(im, (x, y))

        return composite

    # images [[b=1,c,h,w],[b,c,h,w],[b,c,h,w]] -> [[b,c,h,w],[b,c,h,w],[b,c,h,w]]
    # att_output [[b, num_head, q_len, k_len]]
    images_vis = [convert_to_opencv_format(img) for img in images]

    # calculate mean attention
    for i, att in enumerate(att_output):
        att_output[i] = torch.mean(att, dim=1, keepdim=True)

    bs = 0
    num_views = len(images_vis)
    num_layers = len(att_output)
    num_heads = att_output[bs].shape[1]
    patch_size = 16
    num_patches = patch_size * patch_size

    layer_figs = []
    for layer in range(num_layers):
        fig, axes = plt.subplots(num_views, num_heads + 1, figsize=(2 * (num_heads + 1), 2 * num_views))
        for view in range(num_views):
            for i in range(num_heads + 1):

                ax = axes[view, i]
                if i == num_heads:
                    map_img = images_vis[view]
                else:
                    start = num_patches * view
                    end = num_patches * (view + 1)
                    map = att_output[layer][bs, i, 1, start:end].reshape(
                        patch_size, patch_size).float().detach().cpu().numpy()
                    map = (map - np.min(map)) / (np.max(map) - np.min(map) + 1e-6)
                    map_img = overlay(images_vis[view], map)
                ax.imshow(map_img)
                ax.set_title(f'Layer {layer + 1} Head {i + 1}' if i < num_heads else f'RGB{view + 1}')
                ax.axis('off')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_pil = Image.open(buf)
        layer_figs.append(img_pil)
        plt.close(fig)  # 释放内存

    composite = concat_layers_imgs_auto(layer_figs, n_cols=6)

    if wandb:
        return composite

    save_dir = "./attention_maps"
    os.makedirs(save_dir, exist_ok=True)
    composite.save(f"{save_dir}/attn_map.png")   # 最简单
    plt.close()

from sklearn.decomposition import PCA
from PIL import Image
def vis_dino_feat(feat_map, img, patch_size, feat_dim):
    '''
    feat_map: (B,C,H,W)
    img: (B,C,H,W)
    '''
    bs, _, h, w = img.shape

    patch_h = h // patch_size
    patch_w = w // patch_size

    bs = feat_map.shape[0]

    feat_map = feat_map.float()
    img = img.float()
    if img.max() < 255:
        img = img * 255.

    feat_map = feat_map.reshape(bs * patch_h * patch_w, feat_dim).cpu()
    ####################################################################################
    plt.subplot(1, 3, 1)
    img_array = img.permute(0,2,3,1).float().detach().cpu().numpy().astype(np.uint8)
    img_pil = Image.fromarray(img_array[0]).convert('RGB')
    plt.imshow(img_pil)

    pca = PCA(n_components=3)
    pca.fit(feat_map)
    pca_features_original = pca.transform(feat_map)
    pca_features = copy.deepcopy(pca_features_original)
    for i in range(3):
        pca_features[:, i] = ((pca_features[:, i] - pca_features[:, i].min()) /
                                   (pca_features[:, i].max() - pca_features[:, i].min()))

    # pca_features = sklearn.processing.minmax_scale(pca_features)

    pca_features_rgb = pca_features.reshape(bs, patch_h, patch_w, 3)[0]
    plt.subplot(1, 3, 2)
    plt.imshow(pca_features_rgb)

    pca_features_bg = pca_features[:, 0] > 0.95  # from first histogram
    pca_features_fg = ~pca_features_bg

    # 2nd PCA for only foreground patches
    pca.fit(feat_map[pca_features_fg])
    pca_features_left = pca.transform(feat_map[pca_features_fg])

    for i in range(3):
        pca_features_left[:, i] = ((pca_features_left[:, i] - pca_features_left[:, i].min()) /
                                   (pca_features_left[:, i].max() - pca_features_left[:, i].min()))
    #
    pca_features_rgb = pca_features.copy()
    # for black background
    pca_features_rgb[pca_features_bg] = 0
    # new scaled foreground features
    pca_features_rgb[pca_features_fg] = pca_features_left

    # reshaping to numpy image format
    pca_features_rgb = pca_features_rgb.reshape(bs, patch_h, patch_w, 3)[0]
    plt.subplot(1, 3, 3)
    plt.imshow(pca_features_rgb)

    plt.show()
    return pca_features_original.reshape(bs, patch_h, patch_w, 3)[0]

def show_feature_correspondence(feat1, feat2, feat3, img1, img2, img3):
    """
    feat1, feat2, feat3: np.ndarray, shape (H, W, 3) - 特征图
    img1, img2, img3: np.ndarray, shape (H, W, 3) or (H, W) - 原始图片
    """
    img1 = img1.float()
    if img1.max() < 255:
        img1 = img1 * 255.

    img2 = img2.float()
    if img2.max() < 255:
        img2 = img2 * 255.

    img3 = img3.float()
    if img3.max() < 255:
        img3 = img3 * 255.

    img1_array = img1.permute(0, 2, 3, 1).float().detach().cpu().numpy().astype(np.uint8)
    img1_pil = Image.fromarray(img1_array[0]).convert('RGB')
    img2_array = img2.permute(0, 2, 3, 1).float().detach().cpu().numpy().astype(np.uint8)
    img2_pil = Image.fromarray(img2_array[0]).convert('RGB')
    img3_array = img3.permute(0, 2, 3, 1).float().detach().cpu().numpy().astype(np.uint8)
    img3_pil = Image.fromarray(img3_array[0]).convert('RGB')

    H, W, _ = img1_array[0].shape
    h, w, _ = feat1.shape
    ratio = H / h
    def find_correspondence(feat1, feat2, feat3, i, j):
        target = feat1[i, j]  # shape (3,)
        diff_b = feat2 - target  # (H, W, 3)
        dist_b = np.linalg.norm(diff_b, axis=2)  # (H, W)
        b_y, b_x = np.unravel_index(np.argmin(dist_b), dist_b.shape)

        diff_c = feat3 - target
        dist_c = np.linalg.norm(diff_c, axis=2)
        c_y, c_x = np.unravel_index(np.argmin(dist_c), dist_c.shape)

        return (b_y, b_x), (c_y, c_x)

    def feature_to_img(feat):
        # 你可以自定义特征可视化方式
        feat_norm = copy.deepcopy(feat)
        for i in range(3):
            feat_norm[:, :, i] = ((feat_norm[:, :, i] - feat_norm[:, :, i].min())
                               / (feat_norm[:, :, i].max() - feat_norm[:, :, i].min()))
        return feat_norm

    a_img = feature_to_img(feat1)
    b_img = feature_to_img(feat2)
    c_img = feature_to_img(feat3)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 第一行显示原图
    for ax, img, title in zip(axes[0], [img1_pil, img2_pil, img3_pil], ['A Image', 'B Image', 'C Image']):
        im = ax.imshow(img, origin='upper')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # 第二行显示特征
    for ax, img, title in zip(axes[1], [a_img, b_img, c_img], ['A Feature', 'B Feature', 'C Feature']):
        im = ax.imshow(img, origin='upper')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # 在图片和特征图上都加scatter
    scatter_kwargs = dict(s=100, edgecolors='red', facecolors='red', linewidths=2)
    highlight_a_img = axes[0, 0].scatter([], [], **scatter_kwargs)
    highlight_b_img = axes[0, 1].scatter([], [], **scatter_kwargs)
    highlight_c_img = axes[0, 2].scatter([], [], **scatter_kwargs)
    highlight_a_feat = axes[1, 0].scatter([], [], **scatter_kwargs)
    highlight_b_feat = axes[1, 1].scatter([], [], **scatter_kwargs)
    highlight_c_feat = axes[1, 2].scatter([], [], **scatter_kwargs)

    def on_move(event):
        # 只在A的特征图上响应
        if event.inaxes == axes[0,0]:
            if event.xdata is None or event.ydata is None:
                return
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if 0 <= x < W and 0 <= y < H:
                (b_y, b_x), (c_y, c_x) = find_correspondence(feat1, feat2, feat3, int(y / ratio), int(x / ratio))
                # 更新所有scatter
                highlight_a_img.set_offsets([[x, y]])
                highlight_b_img.set_offsets([[b_x * ratio, b_y * ratio]])
                highlight_c_img.set_offsets([[c_x * ratio, c_y * ratio]])
                highlight_a_feat.set_offsets([[x / ratio, y / ratio]])
                highlight_b_feat.set_offsets([[b_x, b_y]])
                highlight_c_feat.set_offsets([[c_x, c_y]])
                fig.canvas.draw_idle()
            else:
                for h in [highlight_a_img, highlight_b_img, highlight_c_img,
                          highlight_a_feat, highlight_b_feat, highlight_c_feat]:
                    h.set_offsets(np.empty((0, 2)))
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.tight_layout()
    plt.show()


def vis_dino_feat_all(feat_maps, imgs, patch_size, feat_dim, save_path='vggt_feat.png'):
    '''
    feat_map: (B,V,P,C)
    img: (B,V,C,H,W)
    '''
    bs, view, _, h, w = imgs.shape

    all_feat_maps = []
    all_imgs = []
    for i in range(view):
        patch_h = h // patch_size
        patch_w = w // patch_size

        # bs = feat_maps.shape[0]
        bs = 1

        feat_maps= feat_maps.float()
        imgs = imgs.float()
        if imgs.max() <= 1:
            imgs[i] = imgs[i] * 255.
        feat_map = feat_maps[:bs,i].reshape(bs * patch_h * patch_w, feat_dim).detach().cpu()
        all_feat_maps.append(feat_map)
        ####################################################################################

        img_array = imgs[:,i].permute(0,2,3,1).float().detach().cpu().numpy().astype(np.uint8)
        all_imgs.append(img_array[0]) #(h,w,3)

    # plt.subplot(3,1,1)
    all_imgs = np.concatenate(all_imgs, axis=1)  # axis=1 表示横向拼接
    img_pil = Image.fromarray(all_imgs).convert('RGB')
    # plt.imshow(img_pil)

    all_feat_maps = np.concatenate(all_feat_maps, axis=0)  # (N*patch_num, feat_dim)

    # ========== 新增：采样view1的n个patch点 ==========

    patch_h = h // patch_size
    patch_w = w // patch_size
    n_patches = patch_h * patch_w
    n_sample = n_patches

    # 随机采样n个patch点
    np.random.seed(42)
    # sampled_idx = np.random.choice(n_patches, n_sample, replace=False)
    # 计算采样点在图片上的坐标
    # sampled_coords = [(idx // patch_w, idx % patch_w) for idx in sampled_idx]  # (row, col)

    # 1. 计算均匀采样的行列坐标
    n_row = patch_h
    n_col = patch_w
    # n_row = int(np.sqrt(n_sample))
    # n_col = int(np.ceil(n_sample / n_row))
    ys = np.linspace(0, patch_h - 1, n_row, dtype=int)
    xs = np.linspace(0, patch_w - 1, n_col, dtype=int)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')  # shape: (n_row, n_col)
    sampled_coords = np.stack([yy.ravel(), xx.ravel()], axis=1)  # (row, col)
    sampled_coords = sampled_coords[:n_sample]  # 只取前n_sample个

    # 2. 计算一维index
    sampled_idx = sampled_coords[:, 0] * patch_w + sampled_coords[:, 1]


    cmap = plt.get_cmap('hsv')
    # colors = cmap(np.linspace(0, 1, n_sample))  # n_sample个颜色
    norm_coords = (sampled_coords - sampled_coords.min(axis=0)) / (sampled_coords.ptp(axis=0) + 1e-8)
    color_vals = 0.5 * norm_coords[:, 0] + 0.5 * norm_coords[:, 1]
    colors = cmap(color_vals)


    # ========== 新增：寻找view2、view3中最接近的patch ==========
    # 取出view1, view2, view3的patch特征
    view1_feats = feat_maps[:bs, 0].reshape(bs, patch_h * patch_w, feat_dim)[0].detach().cpu().numpy()  # (patch_num, feat_dim)
    view2_feats = feat_maps[:bs, 1].reshape(bs, patch_h * patch_w, feat_dim)[0].detach().cpu().numpy()
    view3_feats = feat_maps[:bs, 2].reshape(bs, patch_h * patch_w, feat_dim)[0].detach().cpu().numpy()

    # 记录每个采样点在view2、view3中最接近的patch索引
    all_matched_idx_view2 = []
    all_matched_idx_view3 = []
    all_matched_dist_view2 = []
    all_matched_dist_view3 = []
    thresh = 100#30  # 你可以根据实际情况调整
    top_n = 5
    for idx in sampled_idx:
        feat = view1_feats[idx]
        # 计算距离
        dist2 = np.linalg.norm(view2_feats - feat, axis=1)
        dist3 = np.linalg.norm(view3_feats - feat, axis=1)
        # matched_idx_view2.append(np.argmin(dist2))
        # matched_idx_view3.append(np.argmin(dist3))
        min_idx2 = np.argmin(dist2)
        min_dist2 = dist2[min_idx2]

        min_idx3 = np.argmin(dist3)
        min_dist3 = dist3[min_idx3]
        # print(min_dist2, min_dist3)

        if min_dist2 < thresh:
            all_matched_idx_view2.append(min_idx2)
            all_matched_dist_view2.append(min_dist2)
        else:
            all_matched_idx_view2.append(None)  # 或者-1，表示没有找到
            all_matched_dist_view2.append(float('inf'))

        if min_dist3 < thresh:
            all_matched_idx_view3.append(min_idx3)
            all_matched_dist_view3.append(min_dist3)
        else:
            all_matched_idx_view3.append(None)
            all_matched_dist_view3.append(float('inf'))

    all_matched_dist_view2 = np.array(all_matched_dist_view2)
    # all_matched_idx_view2 = np.array(all_matched_idx_view2, dtype=object)
    topn_indices = np.argsort(all_matched_dist_view2)[:top_n]

    # 新建topn的list
    matched_idx_view2 = [None] * len(all_matched_idx_view2)
    for i in topn_indices:
        matched_idx_view2[i] = all_matched_idx_view2[i]

    matched_dist_view3 = np.array(all_matched_dist_view3)
    # matched_idx_view3 = np.array(matched_idx_view3, dtype=object)
    topn_indices = np.argsort(matched_dist_view3)[:top_n]

    matched_idx_view3 = [None] * len(all_matched_idx_view3)
    for i in topn_indices:
        matched_idx_view3[i] = all_matched_idx_view3[i]


    # 计算对应的坐标
    # matched_coords_view2 = [(idx // patch_w, idx % patch_w) for idx in matched_idx_view2]
    # matched_coords_view3 = [(idx // patch_w, idx % patch_w) for idx in matched_idx_view3]

    fig, ax = plt.subplots(figsize=(20, 20))  # 这里设置图片大小
    # ========== 新增：可视化连线 ==========
    # 计算每个view的左上角在拼接图中的偏移
    offsets = [0, w, 2 * w]  # view1, view2, view3

    plt.subplot(6, 1, 1)
    plt.imshow(img_pil)


    plt.subplot(6, 1, 2)
    plt.imshow(img_pil)


    plt.subplot(6, 1, 3)
    plt.imshow(img_pil)

    plt.subplot(6, 1, 4)
    plt.imshow(img_pil)

    for i in range(n_sample):
        # 采样点
        y1, x1 = sampled_coords[i]
        idx2 = matched_idx_view2[i]
        idx3 = matched_idx_view3[i]

        # patch中心点坐标函数
        def patch_center(y, x, offset):
            return offset + x * patch_size + patch_size // 2, y * patch_size + patch_size // 2

        # 画第一个视角的点
        plt.subplot(6, 1, 2)
        ax = plt.gca()
        c1 = patch_center(y1, x1, offsets[0])
        ax.plot(c1[0], c1[1], 'ro', color=colors[i], markersize=2)

        # 画第二个视角的点和连线
        plt.subplot(6, 1, 3)
        ax = plt.gca()
        if idx2 is not None:
            y2, x2 = idx2 // patch_w, idx2 % patch_w
            c2 = patch_center(y2, x2, offsets[1])
            ax.plot(c1[0], c1[1], 'go', color=colors[i], markersize=6)
            ax.plot(c2[0], c2[1], 'go', color=colors[i], markersize=6)
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], 'r-', color=colors[i], linewidth=1)

        # # 画第三个视角的点和连线
        plt.subplot(6, 1, 4)
        ax = plt.gca()
        if idx3 is not None:
            y3, x3 = idx3 // patch_w, idx3 % patch_w
            c3 = patch_center(y3, x3, offsets[2])
            ax.plot(c1[0], c1 [1], 'bo', color=colors[i], markersize=6)
            ax.plot(c3[0], c3[1], 'bo', color=colors[i], markersize=6)
            ax.plot([c1[0], c3[0]], [c1[1], c3[1]], 'b-', color=colors[i], linewidth=1)

    pca = PCA(n_components=3)
    pca.fit(all_feat_maps)
    pca_features_original = pca.transform(all_feat_maps)
    pca_features = copy.deepcopy(pca_features_original)
    for i in range(3):
        pca_features[:, i] = ((pca_features[:, i] - pca_features[:, i].min()) /
                                   (pca_features[:, i].max() - pca_features[:, i].min()))

    # pca_features = sklearn.processing.minmax_scale(pca_features)

    print(pca_features.shape)
    pca_features_rgb = pca_features.reshape(bs, view, patch_h, patch_w, 3)[0]
    pca_features_rgb = np.concatenate(pca_features_rgb, axis=1)  # axis=1表示宽度方向
    plt.subplot(6,1,5)
    plt.imshow(pca_features_rgb)

    pca_features_bg = pca_features[:, 0] > 0.95  # from first histogram
    pca_features_fg = ~pca_features_bg

    # 2nd PCA for only foreground patches
    pca.fit(all_feat_maps[pca_features_fg])
    pca_features_left = pca.transform(all_feat_maps[pca_features_fg])

    for i in range(3):
        pca_features_left[:, i] = ((pca_features_left[:, i] - pca_features_left[:, i].min()) /
                                   (pca_features_left[:, i].max() - pca_features_left[:, i].min()))
    #
    pca_features_rgb = pca_features.copy()
    # for black background
    pca_features_rgb[pca_features_bg] = 0
    # new scaled foreground features
    pca_features_rgb[pca_features_fg] = pca_features_left

    # reshaping to numpy image format
    pca_features_rgb = pca_features_rgb.reshape(bs, view, patch_h, patch_w, 3)[0]
    pca_features_rgb = np.concatenate(pca_features_rgb, axis=1)  # axis=1表示宽度方向
    plt.subplot(6, 1, 6)

    # 全局关闭所有子图坐标轴
    for ax in plt.gcf().get_axes():
        ax.axis('off')

    plt.imshow(pca_features_rgb)
    plt.show()

    plt.savefig(save_path)
    print("save done")
    return pca_features_original.reshape(bs, patch_h, patch_w * view, 3)[0]


# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import os


def plot_all_joints(action, obs, save_path='all_joints.png', wandb=False):
    N_t, joint_dim = action.shape
    # 自动决定子图的行列数（比如每行4个）
    ncols = min(4, joint_dim)
    nrows = math.ceil(joint_dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False, sharex=True, sharey=True)

    for j in range(joint_dim):
        row, col = divmod(j, ncols)
        ax = axes[row, col]
        ax.plot(action[:, j], label='GT', linewidth=2, marker='o')
        ax.plot(obs[:, j], label='Observation', linestyle='--', linewidth=2, marker='o')
        ax.set_title(f'Joint {j}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Value')
        ax.legend()

    # 去掉多余的空子图
    for j in range(joint_dim, nrows * ncols):
        row, col = divmod(j, ncols)
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)

    if wandb:
        return img_pil

    img_pil.save(save_path)
    print(f"Saved all joint trajectories to {save_path}")

# 用法示例
# gt = np.random.randn(100, 6)
# obs = np.random.randn(100, 6)
# plot_all_joints(gt, obs, save_path='all_joints.png')

def convert_to_opencv_format(image):
    """
    将图像转换为适合OpenCV处理的标准格式

    参数:
        image: 输入图像，可以是numpy数组或PyTorch张量

    返回:
        numpy数组格式的图像，具有以下特性:
        - HWC (高度,宽度,通道) 格式
        - uint8数据类型，像素值范围0-255
        - 连续的内存布局
        - 3通道BGR格式
    """
    # 确保图像是 numpy 数组
    if isinstance(image, torch.Tensor):
        # 如果是 PyTorch 张量，转换为 numpy 数组
        image_np = image.detach().clone()  # 创建副本以避免修改原始张量

        if image_np.dim() == 4:  # BCHW 格式
            image_np = image_np.squeeze(0)  # 移除批次维度

        # 将 CHW 转换为 HWC (适用于 OpenCV)
        if image_np.dim() == 3 and image_np.shape[0] in [1, 3, 4]:
            image_np = image_np.permute(1, 2, 0)

        # 转换为 numpy 数组
        image_np = image_np.float().cpu().numpy()

        image_np = np.clip(image_np, a_min=0, a_max=1)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
    else:
        image_np = image

    # 确保数组是连续的，提高OpenCV处理效率
    image_np = np.ascontiguousarray(image_np)

    # 确保图像是 3 通道
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 1:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    return image_np

def plot_all_images(top_cam, left_cam, right_cam, save_path='all_images.png'):
    #(HISTORY_SIZE, H, W, 3)
    IMG_HISTORY_SIZE = top_cam.shape[0]
    cams = [top_cam, left_cam, right_cam]
    cam_names = ['to_p_cam', 'left_cam', 'right_cam']

    fig, axes = plt.subplots(nrows=3, ncols=IMG_HISTORY_SIZE, figsize=(IMG_HISTORY_SIZE * 2, 6), squeeze=False)

    for row, (cam, name) in enumerate(zip(cams, cam_names)):
        for col in range(IMG_HISTORY_SIZE):
            axes[row, col].imshow(convert_to_opencv_format(cam[col]))
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(f'Frame {col + 1}')
        axes[row, 0].set_ylabel(name, rotation=0, labelpad=40, fontsize=12, va='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved all images to {save_path}")

def plot_all_images_with_depth(top_cam, left_cam, right_cam, top_depth, left_depth, right_depth, save_path='all_images.png'):
    # (HISTORY_SIZE, H, W, 3) for RGB, (HISTORY_SIZE, H, W) for Depth
    IMG_HISTORY_SIZE = top_cam.shape[0]
    cams_rgb = [top_cam, left_cam, right_cam]
    cams_depth = [top_depth, left_depth, right_depth]
    cam_names = ['top_cam', 'left_cam', 'right_cam']

    # 每个摄像头两列：RGB和Depth
    fig, axes = plt.subplots(nrows=3, ncols=IMG_HISTORY_SIZE*2, figsize=(IMG_HISTORY_SIZE * 4, 6), squeeze=False)

    for row, (rgb, depth, name) in enumerate(zip(cams_rgb, cams_depth, cam_names)):
        for col in range(IMG_HISTORY_SIZE):
            # RGB
            axes[row, col*2].imshow(convert_to_opencv_format(rgb[col]))
            axes[row, col*2].axis('off')
            if row == 0:
                axes[row, col*2].set_title(f'RGB {col+1}')
            # Depth, 可选配色(例如'viridis'或'gray')
            axes[row, col*2+1].imshow(depth[col].squeeze(), cmap='viridis') #(H, W) as inputs
            axes[row, col*2+1].axis('off')
            if row == 0:
                axes[row, col*2+1].set_title(f'Depth {col+1}')
        axes[row, 0].set_ylabel(name, rotation=0, labelpad=40, fontsize=12, va='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved all images to {save_path}")

import random
def get_history_indices(step_id: int, history_size: int, interval: int, random_sample: bool = True):
    """
    获取历史帧的索引列表。
    - 如果 random_sample=False，则每隔 interval 取一帧（等间隔）。
    - 如果 random_sample=True，则在每个 interval 区间内随机取一帧。

    Args:
        step_id (int): 当前帧索引
        history_size (int): 历史帧数量
        interval (int): 间隔
        random_sample (bool): 是否在区间内随机采样

    Returns:
        List[int]: 历史帧索引列表,
    """
    indices = []
    for i in range(history_size):
        end = step_id - (history_size - 1 - i) * interval
        if random_sample:
            start = end - interval + 1
            start = max(start, 0)
            end = max(end, 0)
            idx = random.randint(start, end)
        else:
            idx = max(end, 0)
        indices.append(idx)

    # 最后一个current timestep 设置是step_id[..., step_id]
    indices[-1] = step_id
    # print(f"history indices: {indices}")
    # print(step_id)

    #去除最后一帧 current timestep
    return indices[:-1]

def plot_flow(image, flow):
    image = convert_to_opencv_format(image)
    h, w = image.shape[:2]
    flow_pixel = np.zeros_like(flow)
    flow_pixel[:, 0] = flow[:, 0] * w
    flow_pixel[:, 1] = flow[:, 1] * h

    # 轨迹顺序作为颜色映射的值
    color_values = np.arange(len(flow_pixel))

    plt.figure(figsize=(8, 4.5))
    plt.imshow(image)

    # 画点，plasma色带
    sc = plt.scatter(flow_pixel[:, 0], flow_pixel[:, 1], c=color_values, cmap='plasma', s=10)

    # 画连线，plasma色带
    # 构造线段
    segments = np.stack([flow_pixel[:-1], flow_pixel[1:]], axis=1)  # shape: (N-1, 2, 2)
    # 线段颜色
    line_colors = color_values[:-1]  # 每段线用起点的颜色
    lc = LineCollection(segments, cmap='plasma', array=line_colors, linewidths=2)
    plt.gca().add_collection(lc)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('output_flow.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_all_flows(image, gt_flow, pred_flow, instruction, save_path='output_flow.png', wandb=False):
    image = convert_to_opencv_format(image)
    h, w = image.shape[:2]

    def flow_to_pixel(flow):
        flow_pixel = np.zeros_like(flow)
        flow_pixel[:, 0] = flow[:, 0] * w
        flow_pixel[:, 1] = flow[:, 1] * h
        return flow_pixel

    flows = [
        ("GT", gt_flow, 'plasma', 0.6),
        ("Pred", pred_flow, 'viridis', 0.8)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (name, flow, cmap, alpha) in zip(axes, flows):
        ax.imshow(image)
        if len(flow) > 0:
            flow_pixel = flow_to_pixel(flow)
            color_values = np.arange(len(flow_pixel))
            # 画点
            sc = ax.scatter(flow_pixel[:, 0], flow_pixel[:, 1], c=color_values,
                            cmap=cmap, s=10, label=name + " points", alpha=alpha)
            # 画连线
            if len(flow_pixel) > 1:
                segments = np.stack([flow_pixel[:-1], flow_pixel[1:]], axis=1)
                line_colors = color_values[:-1]
                lc = LineCollection(segments, cmap=cmap, array=line_colors,
                                    linewidths=2, alpha=alpha, label=name + " line")
                ax.add_collection(lc)
        # 画 instruction 到每个子图左上角
        # ax.text(5, 20, instruction, fontsize=11, color='black', backgroundcolor='white')
        ax.set_title(name, fontsize=14)
        ax.axis('off')
        ax.legend(loc='lower right', fontsize=10)

    # 或者你只想画在整个图的顶部
    fig.text(0.5, 0.92, instruction, fontsize=14, ha='center', color='black', backgroundcolor='thistle')

    plt.tight_layout()
    if wandb:
        return fig  # 注意要返回fig，不是plt

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

# def loc_to_pts(loc_y, loc_x):
#     # 1. 将 0-1023 范围的值转回 0-1 的归一化坐标
#     norm_y = loc_y / 1023.0
#     norm_x = loc_x / 1023.0
#
#     # # 2. 将归一化坐标乘以图像尺寸得到实际像素坐标
#     # pts_y = (norm_y * height).astype(np.int32)
#     # pts_x = (norm_x * width).astype(np.int32)
#
#     return norm_y, norm_x
def text_to_flow(text):
    pairs = re.findall(r'<loc(\d+)><loc(\d+)>;', text)
    flow = np.array([[int(x), int(y)] for x, y in pairs], dtype=np.float32)
    flow_norm = flow / 1023.0

    return flow_norm


def text_to_norm_point(text):
    # 找到所有 image 部分，如 image1[...]、image2[...]
    image_blocks = re.findall(r'(image\d*)\[([^\]]*)\]', text)

    image_points = {}
    for img_name, coords_str in image_blocks:
        # 匹配所有 <locxxx><locyyy>;
        point_matches = re.findall(r'<loc(\d+)><loc(\d+)>;', coords_str)
        if point_matches:
            # 转为float并归一化
            points = np.array([(int(x), int(y)) for x, y in point_matches], dtype=np.float32)
            points_norm = points / 1023.
        else:
            # points_norm = np.zeros((0, 2), dtype=np.float32)  # 没有点返回空数组
            continue
        image_points[img_name] = points_norm

    return image_points

def text_to_norm_class_point(text):
    # 找到所有 image 部分，如 image1[...]、image2[...]
    image_blocks = re.findall(r'(image\d*)\[([^\]]*)\]', text)

    image_points = {}
    for img_name, coords_str in image_blocks:
        # 匹配所有 类别<locxxx><locyyy>; 例如 left container<loc0481><loc0631>;
        # 注意：类别名可能有空格，且不是<开头
        pattern = r'([^;<]+)<loc(\d+)><loc(\d+)>;'
        point_matches = re.findall(pattern, coords_str)

        if point_matches:
            class_names = []
            points = []
            for cls, x, y in point_matches:
                cls = cls.strip()
                class_names.append(cls)
                points.append([int(x), int(y)])

            points = np.array(points, dtype=np.float32)
            points_norm = points / 1023.

            # 构建 [(class_name, normed_point), ...] 列表
            class_and_points = [
                (cls, point) for cls, point in zip(class_names, points_norm)
            ]
            image_points[img_name] = class_and_points
        else:
            continue

    return image_points

def build_K_from_intrinsics(intrinsics):
    # 转成numpy数组
    if isinstance(intrinsics, torch.Tensor):
        arr = intrinsics.float().detach().cpu().numpy()
    elif isinstance(intrinsics, np.ndarray):
        arr = intrinsics
    elif isinstance(intrinsics, (list, tuple)):
        arr = np.array(intrinsics)
    else:
        raise TypeError(f"Unsupported type: {type(intrinsics)}")

    # 如果本身就是3x3矩阵
    if arr.shape == (3, 3):
        K = arr.astype(np.float32)
    else:
        if arr.shape[0] < 4:
            raise ValueError("Intrinsics must contain at least 4 elements (fx, fy, cx, cy)")
        fx, fy, cx, cy = arr[:4]
        K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float32).reshape(3, 3)
    return K

def euler_to_matrix(rot, degrees=False):
    return Rotation.from_euler('xyz', rot, degrees=degrees).as_matrix()

def project_point(trans, K):
    pt3d = np.array(trans, dtype=np.float32).reshape(3, 1)
    x = K @ pt3d
    x /= x[2]
    return float(x[0][0]), float(x[1][0])

# def draw_bbox_matplot(ax, loc, img_hw, color='#1976D2', label=None):
#     h, w = img_hw
#     if len(loc) == 2:  # 点
#         x = loc[0] * w
#         y = loc[1] * h
#         ax.scatter(x, y, s=70, c=color, marker='o', edgecolors='white', linewidths=1.8, zorder=10)
#         if label:
#             ax.text(x+8, y-8, label, fontsize=12, color='white',
#                     ha='left', va='bottom',
#                     bbox=dict(boxstyle='round,pad=0.19', fc=color, ec='none', alpha=0.85), zorder=12)
#     elif len(loc) == 4:  # 框
#         xmin, ymin, xmax, ymax = loc
#         xmin, xmax = xmin*w, xmax*w
#         ymin, ymax = ymin*h, ymax*h
#         rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2.3,
#                              edgecolor=color, facecolor='none', zorder=9, alpha=0.72)
#         ax.add_patch(rect)
#         if label:
#             ax.text(xmin+8, ymin+11, label, fontsize=12, color='white', ha='left', va='top',
#                     bbox=dict(boxstyle='round,pad=0.18', fc=color, ec='none', alpha=0.7), zorder=11)
#     # else: pass

def draw_bbox_matplot(ax, locs_labels, img_hw, color='#1976D2'):
    h, w = img_hw
    text_list = []
    for i, (loc, label) in enumerate(locs_labels):

        # ---- 点: loc 长度2 ----
        if len(loc) == 2:
            point_color = '#7B1FA2'
            x = loc[0] * w
            y = loc[1] * h
            ax.scatter(x, y, s=40, c=point_color, marker='o', edgecolors='white', linewidths=1.8, zorder=10)
            if label:
                if label and ("trajectory" not in label or i in (0, len(locs_labels)-1)):
                    t = ax.text(x, y, label, fontsize=12, color='white',
                                ha='left', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.2', fc=point_color, ec='none', alpha=0.85), zorder=12)
                    text_list.append(t)
        # ---- 框: loc 长度4 ----
        elif len(loc) == 4:
            xmin, ymin, xmax, ymax = loc
            xmin, xmax = xmin*w, xmax*w
            ymin, ymax = ymin*h, ymax*h
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2.3,
                                 edgecolor=color, facecolor='none', zorder=9, alpha=0.72)
            ax.add_patch(rect)
            if label:
                label_offset = 6  # 缝隙可调整
                ax.text(xmin, ymin - label_offset, label, fontsize=12, color='white', ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.18', fc=color, ec='none', alpha=0.7), zorder=11)
        # ---- 其他情况: 按点画 ----
        else:
            point_color = '#7B1FA2'
            # 假设loc里面有两个元素（否则你需设计容错）
            # 如果长度不是2，需要选前两个分量作为点坐标
            # 你可以用更多的自定义策略，比如取前两个分量
            try:
                x = loc[0] * w
                y = loc[1] * h
            except Exception as e:
                continue  # 跳过错误数据
            ax.scatter(x, y, s=70, c=point_color, marker='o', edgecolors='white', linewidths=1.8, zorder=10)

            if label:
                t = ax.text(x, y, label, fontsize=12, color='white', ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2', fc=point_color, ec='none', alpha=0.85), zorder=12)
                text_list.append(t)
    # 只有点的label自动调整
    if text_list:
        adjust_text(
            text_list,
            ax=ax,
            force_arrow=True,
            arrowprops=dict(arrowstyle='-', color='#7B1FA2')
        )

def draw_axes_matplot(ax, trans, rot, K, length=0.02, colors=['#FF5252','#76FF03','#2979FF'], alpha=0.92):
    center_2d = project_point(trans, K)
    R = euler_to_matrix(rot)
    axes_dirs = np.eye(3)*length
    for i in range(3):
        end_3d = np.array(trans) + R @ axes_dirs[i]
        end_2d = project_point(end_3d, K)
        ax.plot([center_2d[0], end_2d[0]], [center_2d[1], end_2d[1]], colors[i], lw=3.0, alpha=alpha, zorder=19)
    ax.scatter([center_2d[0]], [center_2d[1]], c='w', s=21, lw=0.7, alpha=alpha, zorder=21)

def draw_3d_bbox_matplot(ax, trans, rot, size, K, color='#8E24AA', linewidth=2, fill=True, fill_alpha=0.5, ground_dir=np.array([0,-1,0])):
    # 获取立方体的长宽高
    l, w, h = size

    # 构造8个立方体角点（局部坐标系原点在立方体中心）
    corners_local = np.array([[l / 2, w / 2, h / 2],
                              [l / 2, -w / 2, h / 2],
                              [-l / 2, -w / 2, h / 2],
                              [-l / 2, w / 2, h / 2],
                              [l / 2, w / 2, -h / 2],
                              [l / 2, -w / 2, -h / 2],
                              [-l / 2, -w / 2, -h / 2],
                              [-l / 2, w / 2, -h / 2]])

    # 旋转+平移，把角点从局部坐标系变换到世界坐标系
    R = euler_to_matrix(rot)  # 欧拉角转旋转矩阵
    world_corners = np.dot(corners_local, R.T) + trans

    # 将世界坐标系下的八个角点投影到像素坐标
    pts_2d = np.array([project_point(pt, K) for pt in world_corners])

    # 立方体的6个面，每面4个点
    cube_faces = [
        [0,1,2,3],  # 顶面(+z)
        [4,5,6,7],  # 底面(-z)
        [0,4,5,1],  # 侧面(+x)
        [1,5,6,2],  # 侧面(-y)
        [2,6,7,3],  # 侧面(-x)
        [3,7,4,0],  # 侧面(+y)
    ]

    # 找和地面方向ground_dir最接近的那个面
    max_dot = -np.inf
    best_face_idx = None
    ground_dir = np.asarray(ground_dir, dtype=np.float32)
    ground_dir = ground_dir / np.linalg.norm(ground_dir)
    for fi, face in enumerate(cube_faces):
        # 计算面法线
        p0, p1, p2 = [world_corners[i] for i in face[:3]]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        normal = normal / (np.linalg.norm(normal) + 1e-7)
        dot = normal @ ground_dir  # 法线与地面的方向内积
        if dot > max_dot:  # 保留最接近地面的（点积最大）
            max_dot = dot
            best_face_idx = fi

    # 拿到要填充的面的4个角点的2D投影坐标
    base_idx = cube_faces[best_face_idx]
    base_pts2d = pts_2d[base_idx]

    # 填充底面
    if fill:
        # 用Polygon加入面到ax
        poly = Polygon(base_pts2d, closed=True, facecolor=color, edgecolor=None, alpha=fill_alpha, zorder=8)
        ax.add_patch(poly)

    # 画立方体的所有边
    edges = [
        [0,1],[1,2],[2,3],[3,0],   # 顶面边
        [4,5],[5,6],[6,7],[7,4],   # 底面边
        [0,4],[1,5],[2,6],[3,7],   # 竖直边
    ]
    for a,b in edges:
        ax.plot([pts_2d[a,0], pts_2d[b,0]], [pts_2d[a,1], pts_2d[b,1]],
                color, lw=linewidth, alpha=0.9, zorder=9)

    # 画八个角点作为散点（方便观察结构）
    ax.scatter(pts_2d[:,0], pts_2d[:,1],
               c=color, s=20, lw=0.8, edgecolors='white', zorder=11, alpha=0.8)


def draw_gripper_2d_with_arm(ax, trans, rot, size, K,
                             arm_len=0.08, arm_thick=0.019,
                             color='dodgerblue', arm_color='dimgray',
                             linewidth=2, fill=True, fill_alpha=0.5):
    """
    在matplotlib ax上画一个抓手（U形指+臂），并投影到2D。
    ax      : matplotlib 2D轴对象
    trans   : (3,) np.array, gripper在世界坐标中的xyz位置
    rot     : (3,)欧拉角(xyz) 或 (3,3)旋转矩阵
    size    : [width_x, thickness_y, finger_length_z]，三个抓手参数
    K       : (3,3) 投影矩阵（内参矩阵）
    arm_len : float, 臂长度（世界单位m）
    arm_thick: float, 臂的粗细（方形截面边长，世界单位m）
    color   : gripper颜色
    arm_color: 臂颜色
    linewidth: 线宽
    fill, fill_alpha: 是否填充抓手/臂面及透明度
    返回所有点投影后的(x, y)像素坐标（np.array，shape=(总点数, 2)）
    坐标系约定：gripper坐标，front=z, right=x, down=y
    """
    width, thick, finger_len = size

    # ========== 1. 构造本地坐标系下的所有关键点 ==========
    # ---- 基座 ----
    base = np.array([
        [ width/2,  thick/2, 0],   # 右上
        [ width/2, -thick/2, 0],   # 右下
        [-width/2, -thick/2, 0],   # 左下
        [-width/2,  thick/2, 0],   # 左上
    ])

    # ---- 左抓指 ----
    left_finger = np.array([
        [-width/2,  thick/2, 0],                    # 左指根 上
        [-width/2,  thick/2, finger_len],           # 左指尖 上
        [-width/2, -thick/2, finger_len],           # 左指尖 下
        [-width/2, -thick/2, 0],                    # 左指根 下
    ])
    # ---- 右抓指 ----
    right_finger = np.array([
        [ width/2,  thick/2, 0],                    # 右指根 上
        [ width/2,  thick/2, finger_len],           # 右指尖 上
        [ width/2, -thick/2, finger_len],           # 右指尖 下
        [ width/2, -thick/2, 0],                    # 右指根 下
    ])

    # ---- 臂（方形杆，底面在gripper base中心，顶面在 -z 方向 arm_len 处） ----
    arm = np.array([
        [ arm_thick/2,  arm_thick/2,     0],              # 底面右上
        [ arm_thick/2, -arm_thick/2,     0],              # 底面右下
        [-arm_thick/2, -arm_thick/2,     0],              # 底面左下
        [-arm_thick/2,  arm_thick/2,     0],              # 底面左上

        [ arm_thick/2,  arm_thick/2, -arm_len],           # 顶面右上
        [ arm_thick/2, -arm_thick/2, -arm_len],           # 顶面右下
        [-arm_thick/2, -arm_thick/2, -arm_len],           # 顶面左下
        [-arm_thick/2,  arm_thick/2, -arm_len],           # 顶面左上
    ])

    # 所有点拼起来
    all_points = np.vstack([base, left_finger, right_finger, arm])   # shape=(20,3)

    # ========== 2. 旋转+平移 → 世界坐标 ==========
    if rot.shape == (3,):
        # 使用欧拉角，xyz方向（rad），你可以改用你的euler_to_matrix
        from scipy.spatial.transform import Rotation as R
        Rmat = R.from_euler('xyz', rot, degrees=False).as_matrix()
    else:
        Rmat = rot
    world_points = np.dot(all_points, Rmat.T) + trans    # shape=(20,3)

    # ========== 3. 投影到2D像素坐标 ==========
    def project_point(pt, K):
        pt_cam = pt   # 假设世界-相机不变换
        uvw = K @ pt_cam
        return uvw[:2] / uvw[2]
    pts_2d = np.array([project_point(p, K) for p in world_points])

    # ========== 4. 可视化抓手与臂 ==========
    # ---- (a) 基座 ----
    base2d = pts_2d[0:4]
    poly_base = Polygon(base2d, closed=True,
                        facecolor=color if fill else 'none',
                        edgecolor=color, alpha=fill_alpha, lw=linewidth, zorder=5)
    ax.add_patch(poly_base)
    # ---- (b) 左右抓指 ----
    left2d = pts_2d[4:8]
    right2d = pts_2d[8:12]
    poly_left = Polygon(left2d, closed=True,
                        facecolor=color if fill else 'none',
                        edgecolor=color, alpha=fill_alpha, lw=linewidth, zorder=6)
    poly_right = Polygon(right2d, closed=True,
                         facecolor=color if fill else 'none',
                         edgecolor=color, alpha=fill_alpha, lw=linewidth, zorder=6)
    ax.add_patch(poly_left)
    ax.add_patch(poly_right)

    # ---- (c) 臂（两面、边线） ----
    arm2d = pts_2d[12:20]
    # 顶面（负z方向端点）
    poly_arm_top = Polygon(arm2d[4:8], closed=True,
                           facecolor=arm_color if fill else 'none',
                           edgecolor=arm_color, alpha=fill_alpha*0.7, lw=linewidth, zorder=3)
    # 底面（连接gripper base）
    poly_arm_base = Polygon(arm2d[0:4], closed=True,
                            facecolor=arm_color if fill else 'none',
                            edgecolor=arm_color, alpha=fill_alpha*0.6, lw=linewidth, zorder=3)
    ax.add_patch(poly_arm_top)
    ax.add_patch(poly_arm_base)
    # 臂的边线
    for i in range(4):
        j = (i+1)%4
        # 底面
        ax.plot([arm2d[i,0], arm2d[j,0]], [arm2d[i,1], arm2d[j,1]],
                 arm_color, lw=linewidth)
        # 顶面
        ax.plot([arm2d[i+4,0], arm2d[(j+4)%8,0]], [arm2d[i+4,1], arm2d[(j+4)%8,1]],
                 arm_color, lw=linewidth)
        # 四条侧边
        ax.plot([arm2d[i,0], arm2d[i+4,0]], [arm2d[i,1], arm2d[i+4,1]],
                 arm_color, lw=linewidth)
    # ---- 可视化所有关键点 ----
    ax.scatter(pts_2d[:,0], pts_2d[:,1], c=color, s=18, edgecolor='white', lw=0.6, zorder=9)

    # ---- 抓手基座和爪边线 ----
    # base边
    for i in range(4):
        j = (i+1)%4
        ax.plot([base2d[i,0], base2d[j,0]],
                [base2d[i,1], base2d[j,1]], color, lw=linewidth)
    # left/right finger
    for finger2d in [left2d, right2d]:
        for i in range(4):
            j = (i+1)%4
            ax.plot([finger2d[i,0], finger2d[j,0]],
                    [finger2d[i,1], finger2d[j,1]], color, lw=linewidth)
    # 连gripper base到两个指根
    for bi, fi in zip([0,3], [0,3]):   # base[0], left[0]; base[3], left[3]
        ax.plot([base2d[bi,0], left2d[fi,0]], [base2d[bi,1], left2d[fi,1]], color, lw=linewidth)
    for bi, fi in zip([1,2], [0,3]):   # base[1], right[0]; base[2], right[3]
        ax.plot([base2d[bi,0], right2d[fi,0]], [base2d[bi,1], right2d[fi,1]], color, lw=linewidth)

    return pts_2d


def visualize_2d_3d_all(images, gt_res, pred_res, intrinsics=None, instruction='', save_path='output_point.png', vis_2d=True, vis_3d=True, wandb=False):
    '''
    images: dict, {img_name: img (np.uint8, RGB, shape=(H,W,3)/(torch.Tensor, RGB, shape=3,H,W))}
    gt_res/pred_res: dict, {img_name: [obj, ...]}, 每个obj: loc/trans/rot/size/class
    intrinsics: dict, {img_name: intr (matrix or [fx,fy,cx,cy])}
    '''
    img_names = list(images.keys())
    N = len(img_names)
    fig, axs = plt.subplots(2, N, figsize=(4 * N, 8))
    for row, mode in enumerate(['gt','pred']):
        obj_dict = gt_res if mode=='gt' else pred_res
        for col, img_name in enumerate(img_names):
            img = convert_to_opencv_format(images[img_name])
            if img.shape[-1]==3 and img.dtype==np.uint8:
                img_rgb = img.copy()
            else:
                img_rgb = img
            h, w = img_rgb.shape[:2]

            if intrinsics[img_name] is not None:
                K = build_K_from_intrinsics(intrinsics[img_name])

            ax = axs[row, col] if N > 1 else axs[row]
            ax.imshow(img_rgb)
            objs = obj_dict.get(img_name, [])
            # -------叠加画所有特征
            # for obj in objs:
            #     if obj is None or 'loc' not in obj: continue
            #     label = obj.get('class', None)
            #     draw_bbox_matplot(ax, obj['loc'], (h,w), color='#1976D2', label=label)
            loc_label_list = [(obj['loc'], obj.get('class', None)) for obj in objs
                              if obj is not None and obj['loc'] is not None]
            if vis_2d:
                draw_bbox_matplot(ax, loc_label_list, (h, w), color='#1976D2')

            if vis_3d:
                for idx, obj in enumerate(objs):
                    if obj is None: continue
                    if obj['trans'] is not None and obj['rot'] is not None:
                        # if len(objs) > 1:
                        #     alpha = 0.25 + 0.75 * (1 - idx / (len(objs) - 1))
                        # else:
                        #     alpha = 1.0
                        alpha = 1.0
                        #TODO: length=obj["size"].max() * 0.5
                        draw_axes_matplot(ax, obj['trans'], obj['rot'], K, length=0.1, alpha=alpha)
                for obj in objs:
                    if obj is None: continue
                    if obj['trans'] is not None and obj['rot'] is not None and obj['size'] is not None:
                        if "grasp" in instruction:
                            draw_gripper_2d_with_arm(ax, obj['trans'], obj['rot'], obj['size'], K)
                        else:
                            draw_3d_bbox_matplot(ax, obj['trans'], obj['rot'], obj['size'], K)
            ax.set_title(f"{img_name} - {mode.upper()}", fontsize=14, backgroundcolor='#F3E5F5' if mode=='gt' else '#E1F5FE', color='#222', pad=10)
            ax.axis('off')
            ax.set_xlim(0, img.shape[1] - 1)
            ax.set_ylim(img.shape[0] - 1, 0)

    # 大标题
    fig.suptitle(instruction, fontsize=18, color='white', backgroundcolor='#7B1FA2', y=0.99)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)

    if wandb:
        return img_pil

    img_pil.save(save_path)

def visualize_traj(images, gt_res, pred_res, intrinsics=None, instruction='', save_path='output_point.png', wandb=False):
    '''
    images: dict, {img_name: img (np.uint8, RGB, shape=(H,W,3)/(torch.Tensor, RGB, shape=3,H,W))}
    gt_res/pred_res: dict, {obj: [...]}, 每个obj: loc/trans/rot/size/class
    intrinsics: dict, {img_name: intr (matrix or [fx,fy,cx,cy])}
    '''
    img_names = list(images.keys())
    N = len(img_names)
    fig, axs = plt.subplots(2, N, figsize=(4 * N, 8))
    for row, mode in enumerate(['gt','pred']):
        obj_dict = gt_res if mode=='gt' else pred_res
        for col, img_name in enumerate(img_names):
            img = convert_to_opencv_format(images[img_name])
            if img.shape[-1]==3 and img.dtype==np.uint8:
                img_rgb = img.copy()
            else:
                img_rgb = img
            h, w = img_rgb.shape[:2]

            if intrinsics[img_name] is not None:
                K = build_K_from_intrinsics(intrinsics[img_name])

            ax = axs[row, col] if N > 1 else axs[row]
            ax.imshow(img_rgb)
            objs = obj_dict.get(img_name, [])
            # -------叠加画所有特征
            for idx, obj in enumerate(objs):
                if obj is None: continue
                if obj['trans'] is not None and obj['rot'] is not None:
                    if len(objs) > 1:
                        alpha = 0.25 + 0.75 * (1 - idx / (len(objs) - 1))
                    else:
                        alpha = 1.0
                    draw_axes_matplot(ax, obj['trans'], obj['rot'], K, length=0.1, alpha=alpha)

                    #TODO: length=obj["size"].max() * 0.5
                    # draw_axes_matplot(ax, obj['trans'], obj['rot'], K, length=0.1)
            ax.set_title(f"{img_name} - {mode.upper()}", fontsize=14, backgroundcolor='#F3E5F5' if mode=='gt' else '#E1F5FE', color='#222', pad=10)
            ax.axis('off')

    # 大标题
    fig.suptitle(instruction, fontsize=18, color='white', backgroundcolor='#7B1FA2', y=0.99)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)

    if wandb:
        return img_pil

    img_pil.save(save_path)

def visualize_views(
    image_dict,    # {name: tensor}, shape (B, History, C, H, W), C=3
    depth_dict,    # {name: tensor}, shape (B, History, 2, H, W)，通常第0通道是depth
    ray_dict,      # {name: tensor}, shape (B, History, 3, H, W)
    mask_dict=None,   # 可选, {name: tensor}, shape (B, History, H, W) or (H, W)
    figsize=(12, 6),
    show_channel_labels=True,
    save_path="output_depth_ray.png"
):
    """
    一行可视化每个view的rgb、depth、rays(normal map)，对比显示。
    """
    names = list(image_dict.keys())
    num_views = len(names)
    num_types = 3   # rgb, depth, ray
    fig, axs = plt.subplots(num_views, num_types, figsize=(figsize[0], figsize[1]*num_views))

    # 如果只有1个view，axis shape修正
    if num_views == 1:
        axs = axs[None, :]
    elif num_types == 1:
        axs = axs[:, None]

    for i, name in enumerate(names):
        # RGB 图像
        img = image_dict[name]  # (C,H,W)
        img = img.permute(1, 2, 0).float().cpu().numpy()        # (H,W,C)
        img = img.clip(0, 1)                            # 0-1

        axs[i,0].imshow(img)
        axs[i,0].set_title(f"{name}\nRGB" if show_channel_labels else name)
        axs[i,0].axis('off')

        # Depth (通常选第0通道，或你用 [batch_idx, history_idx, 0] )
        depth_map = depth_dict[name][0].float().cpu().numpy()  # (H,W)
        # 显示为灰度图
        axs[i,1].imshow(depth_map, cmap='plasma')  # 或 'viridis', 'magma'
        axs[i,1].set_title(f"Depth", fontsize=10)
        axs[i,1].axis('off')

        # Rays (normal map风格)
        rays = ray_dict[name]                             # (3,H,W)
        rays = rays.permute(1,2,0)                        # (H,W,3)
        norm = torch.linalg.norm(rays, dim=2, keepdim=True) + 1e-6
        rays_rgb = (rays / norm + 1) / 2                  # 映射到[0,1]
        rays_rgb = rays_rgb.clamp(0,1).float().cpu().numpy()

        # mask
        if mask_dict is not None and name in mask_dict:
            mask = mask_dict[name]
            if mask.ndim == 4:
                mask_show = mask
            else:
                mask_show = mask
            rays_rgb[mask_show.cpu().numpy()==0] = 0
            img[mask_show.cpu().numpy()==0] = 0
            depth_map[mask_show.cpu().numpy()==0] = 0

        axs[i,2].imshow(rays_rgb)
        axs[i,2].set_title(f"Ray(Normal Map)", fontsize=10)
        axs[i,2].axis('off')

    plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)
    #
    # if wandb:
    #     return img_pil
    img_pil.save(save_path)

def plot_all_class_points(images, gt_image_points, pred_image_points, instruction, save_path='output_point.png', wandb=False):
    # images: [(b, h, w, c)]
    # gt_image_points: {img_name: [(class_name, norm_point), ..., (class_name, norm_point)]}
    name2idx = {f"image{i}": i for i in range(len(images))}
    N = len(images)

    fig, axes = plt.subplots(2, N, figsize=(4 * N, 8))
    if N == 1:
        axes = np.array(axes).reshape(2, 1)

    plasma = plt.get_cmap('plasma')
    viridis = plt.get_cmap('viridis')

    bs = 0
    for i in range(N):
        img_name = f"image{i}"
        idx = name2idx[img_name]
        img = images[idx][bs]
        img = convert_to_opencv_format(img)

        h, w = img.shape[:2]

        # --------- GT -----------
        axes[0, i].imshow(img)
        texts_gt = []
        if img_name in gt_image_points:
            gt_pairs = gt_image_points[img_name]
            num_gt = len(gt_pairs)
            gt_colors = plasma(np.linspace(0, 1, num_gt))
            for idx_point, (cls, point) in enumerate(gt_pairs):
                x, y = point[0] * w, point[1] * h
                # color = gt_colors[idx_point]
                color = '#7B1FA2'
                axes[0, i].scatter([x], [y], s=40, c=[color], marker='o', edgecolors='white', linewidths=1, zorder=10)
                # 文本背景色设为和点一样
                t = axes[0, i].text(
                    x, y, cls,
                    fontsize=11, color='white', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc=color, ec='none', alpha=1.0),
                    zorder=4
                )
                texts_gt.append(t)
            # 自动调整文本
            # only_move = {'points': 'y', 'text': 'y'}
            adjust_text(texts_gt,
                        ax=axes[0, i],
                        # only_move = {'points': 'y', 'text': 'y'},
                        arrowprops=dict(arrowstyle='-', color=color, lw=1, alpha=1.0, zorder=2),
                        force_arrow=True,
                        # expand_points=(2, 2),  # 挪远标签
                        # expand_text=(1.25, 1.2),  # 标签之间更疏
                        # force_text=0.2,  # 增加调整强度
                        # only_move={'text': 'xy'},  # 只移动文本
                        # lim=1000  # 增加迭代数量
            )

        axes[0, i].set_title(f"{img_name} - GT")
        axes[0, i].axis('off')

        # --------- Pred ----------
        axes[1, i].imshow(img)
        texts_pred = []
        if img_name in pred_image_points:
            pred_pairs = pred_image_points[img_name]
            num_pred = len(pred_pairs)
            pred_colors = viridis(np.linspace(0, 1, num_pred))
            for idx_point, (cls, point) in enumerate(pred_pairs):
                x, y = point[0] * w, point[1] * h
                # color = pred_colors[idx_point]
                color = '#7B1FA2'
                axes[1, i].scatter([x], [y], s=40, c=[color], marker='o', edgecolors='white', linewidths=1, zorder=10)
                t = axes[1, i].text(
                    x, y, cls,
                    fontsize=11, color='white', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc=color, ec='none', alpha=1.0),
                    zorder = 4
                )
                texts_pred.append(t)
            adjust_text(texts_pred,
                        ax=axes[1, i],
                        # only_move = {'points': 'y', 'text': 'y'},
                        arrowprops=dict(arrowstyle='-', color=color, lw=1, alpha=1.0, zorder=2),
                        force_arrow=True,
                        # expand_points=(2, 2),  # 挪远标签
                        # expand_text=(1.25, 1.2),  # 标签之间更疏
                        # force_text=0.2,  # 增加调整强度
                        # only_move={'text': 'xy'},  # 只移动文本
                        # lim=1000  # 增加迭代数量
            )

        axes[1, i].set_title(f"{img_name} - Pred")
        axes[1, i].axis('off')

    fig.text(0.5, 0.92, instruction, fontsize=14, ha='center', color='black', backgroundcolor='thistle')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)

    if wandb:
        return img_pil

    img_pil.save(save_path)


def batch_overlay_gaussian_mask(images, masks, alpha=0.5, color_map=cv2.COLORMAP_JET):
    """
    images: [num, h, w, 3]/[num, 3, h, w] uint8 (BGR)
    masks:  [num, h, w]/ [num, 1, h, w] float/uint8，高斯mask，取值0~1或0~255
    alpha:  叠加透明度
    返回:   [num, h, w, 3] uint8, mask可视化后的图片
    """

    # convert to [num, h, w]
    if masks.ndim == 3:
        pass
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:,0]
    else:
        raise ValueError("shape error")

    if isinstance(masks, torch.Tensor):
        masks = masks.float().cpu().numpy()

    num = images.shape[0]
    overlays = []

    for i in range(num):
        img = convert_to_opencv_format(images[i])  # [h, w, 3]
        mask = masks[i]  # [h, w]
        # 归一化mask到0~255
        mask_norm = mask
        if mask.max() > 0:
            mask_norm = (mask / mask.max() * 255).astype(np.uint8)
        else:
            mask_norm = mask.astype(np.uint8)

        mask_color = cv2.applyColorMap(255 - mask_norm, color_map)  # [h, w, 3]
        overlaid = cv2.addWeighted(img, 1 - alpha, mask_color, alpha, 0)
        overlays.append(overlaid)
    return np.stack(overlays, axis=0)
