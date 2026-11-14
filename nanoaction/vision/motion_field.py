# motion_field.py
import math
from collections import deque
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# ----------------------------
# Utilities
# ----------------------------

def to_gray_torch(frame_bgr_uint8: torch.Tensor) -> torch.Tensor:
    """
    frame_bgr_uint8: [H,W,3] uint8 (0..255), BGR or RGB both ok (weights average)
    returns: [1,1,H,W] float32 in [0,1]
    """
    if frame_bgr_uint8.dtype != torch.uint8:
        raise ValueError("Expected uint8 frame")
    f = frame_bgr_uint8.float() / 255.0  # [H,W,3]
    if f.shape[-1] == 3:
        # approximate luminance, indifferent to channel order for MVP
        gray = (0.299 * f[..., 2] + 0.587 * f[..., 1] + 0.114 * f[..., 0])  # [H,W]
    else:
        gray = f[..., 0]
    gray = gray.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return gray


def build_pyramid(img: torch.Tensor, levels: int) -> list:
    """
    img: [1,1,H,W], float
    returns list coarse->fine
    """
    pyr = [img]
    for _ in range(1, levels):
        img = F.avg_pool2d(img, kernel_size=2, stride=2, padding=0)
        pyr.append(img)
    return pyr[::-1]  # coarse first


def normalized_base_grid(H: int, W: int, device, dtype=torch.float32) -> torch.Tensor:
    """
    returns grid [1,H,W,2] in normalized coords [-1,1], x then y
    """
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1,H,W,2]
    return grid


def pixels_to_norm(dx: torch.Tensor, dy: torch.Tensor, W: int, H: int):
    # Convert pixel displacement to normalized grid displacement
    return 2.0 * dx / (W - 1), 2.0 * dy / (H - 1)


def flow_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    img: [1,1,H,W], flow: [1,2,H,W] in pixel units (dx, dy)
    returns warped img2 at positions of img1
    """
    B, C, H, W = img.shape
    base = normalized_base_grid(H, W, img.device, img.dtype)  # [1,H,W,2]
    dxn, dyn = pixels_to_norm(flow[:, 0], flow[:, 1], W, H)
    grid = torch.stack([base[..., 0] + dxn, base[..., 1] + dyn], dim=-1)  # [1,H,W,2]
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)


# ----------------------------
# Dense flow via correlation volume (coarse-to-fine)
# ----------------------------

@torch.inference_mode()
def dense_flow_correlate(
    im1: torch.Tensor, im2: torch.Tensor,
    levels: int = 2,  # 2 or 3 suggested
    search_radius: int = 3,  # +/- r pixels per level
    patch_agg: int = 3,  # box filter kernel for cost smoothing
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    im1, im2: [1,1,H,W] (float32 [0,1])
    returns flow12: [1,2,H,W] in pixel units (dx, dy) from im1 -> im2
    """
    device = im1.device
    # convert to working dtype for speed
    im1 = im1.to(dtype)
    im2 = im2.to(dtype)

    pyr1 = build_pyramid(im1, levels=levels)
    pyr2 = build_pyramid(im2, levels=levels)

    flow = None
    for lvl in range(levels):
        I1 = pyr1[lvl]  # [1,1,h,w]
        I2 = pyr2[lvl]
        B, C, H, W = I1.shape
        base = normalized_base_grid(H, W, device, dtype)

        # upsample previous flow
        if flow is None:
            flow = torch.zeros((B, 2, H, W), device=device, dtype=dtype)
        else:
            flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True) * 2.0  # scale pixels

        # build offsets
        offs = []
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                dxn, dyn = pixels_to_norm(
                    torch.full((1, H, W), dx, device=device, dtype=dtype),
                    torch.full((1, H, W), dy, device=device, dtype=dtype),
                    W, H
                )
                offs.append(torch.stack([dxn, dyn], dim=-1))  # [1,H,W,2]
        K = len(offs)
        offs = torch.stack(offs, dim=0)  # [K,1,H,W,2]

        # flow to normalized
        dxn, dyn = pixels_to_norm(flow[:, 0], flow[:, 1], W, H)
        flow_norm = torch.stack([dxn, dyn], dim=-1)  # [1,H,W,2]

        # replicate and sample
        grid = base + flow_norm  # [1,H,W,2]
        gridK = grid.unsqueeze(0) + offs  # [K,1,H,W,2]
        I2K = F.grid_sample(I2.expand(K, -1, -1, -1), gridK.squeeze(1),
                            mode='bilinear', padding_mode='border', align_corners=True)  # [K,1,H,W]
        I1K = I1.expand(K, -1, -1, -1)  # [K,1,H,W]

        # cost volume (SSD, aggregated)
        diff = (I1K - I2K) ** 2  # [K,1,H,W]
        if patch_agg > 1:
            diff = F.avg_pool2d(diff, kernel_size=patch_agg, stride=1, padding=patch_agg // 2)
        cost = diff.squeeze(1)  # [K,H,W]

        # argmin over K
        best_idx = torch.argmin(cost, dim=0)  # [H,W]
        # map idx -> (dx, dy)
        disp = torch.zeros((2, H, W), device=device, dtype=dtype)
        k = 0
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                mask = (best_idx == k)
                disp[0][mask] = dx
                disp[1][mask] = dy
                k += 1

        flow = flow + disp.unsqueeze(0)  # refine at this level

    return flow.to(torch.float32)  # [1,2,H,W] in pixels


# ----------------------------
# Homography via Torch DLT + RANSAC
# ----------------------------

def _normalize_points(xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hartley normalization.
    xy: [N,2]
    returns: xy_n [N,2], T [3,3]
    """
    mean = xy.mean(dim=0)
    d = torch.sqrt(((xy - mean) ** 2).sum(dim=1)).mean()
    s = math.sqrt(2.0) / (d + 1e-8)
    T = torch.tensor([[s, 0, -s * mean[0]],
                      [0, s, -s * mean[1]],
                      [0, 0, 1.0]], device=xy.device, dtype=xy.dtype)
    ones = torch.ones((xy.shape[0], 1), device=xy.device, dtype=xy.dtype)
    xy1 = torch.cat([xy, ones], dim=1).T  # [3,N]
    xy_n = (T @ xy1).T[:, :2]
    return xy_n, T


def _dlt_homography(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    p1, p2: [M,2] (M>=4, normalized)
    returns H (3x3) normalized-space
    """
    M = p1.shape[0]
    x, y = p1[:, 0], p1[:, 1]
    u, v = p2[:, 0], p2[:, 1]
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    A1 = torch.stack([ -x, -y, -ones, zeros, zeros, zeros, x*u, y*u, u ], dim=1)
    A2 = torch.stack([ zeros, zeros, zeros, -x, -y, -ones, x*v, y*v, v ], dim=1)
    A = torch.cat([A1, A2], dim=0)  # [2M,9]
    # solve Ah=0 via SVD
    _, _, V = torch.linalg.svd(A, full_matrices=False)
    h = V[-1, :]  # smallest singular vector
    H = h.view(3, 3)
    return H


@torch.inference_mode()
def fit_homography_ransac(
    p1: torch.Tensor, p2: torch.Tensor,
    iters: int = 256, thresh: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    p1, p2: [N,2] pixel coordinates
    returns: H (3x3), inlier_mask [N] bool
    """
    device = p1.device
    N = p1.shape[0]
    best_inliers = None
    best_H = None
    if N < 4:
        H = torch.eye(3, device=device)
        inliers = torch.zeros(N, dtype=torch.bool, device=device)
        return H, inliers

    # Normalization
    p1n, T1 = _normalize_points(p1)
    p2n, T2 = _normalize_points(p2)

    for _ in range(iters):
        idx = torch.randperm(N, device=device)[:4]
        Hn = _dlt_homography(p1n[idx], p2n[idx])

        # Denormalize
        H = torch.linalg.inv(T2) @ Hn @ T1

        # Reprojection error
        ones = torch.ones((N, 1), device=device)
        P1 = torch.cat([p1, ones], dim=1).T  # [3,N]
        P2p = (H @ P1)  # [3,N]
        P2p = (P2p[:2] / (P2p[2:3] + 1e-8)).T  # [N,2]
        err = torch.linalg.norm(P2p - p2, dim=1)  # [N]
        inliers = err < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_H = H

    if best_inliers is None:
        best_H = torch.eye(3, device=device)
        best_inliers = torch.zeros(N, dtype=torch.bool, device=device)
        return best_H, best_inliers

    # refine with all inliers
    in_idx = torch.where(best_inliers)[0]
    if in_idx.numel() >= 4:
        p1n, T1 = _normalize_points(p1[in_idx])
        p2n, T2 = _normalize_points(p2[in_idx])
        Hn = _dlt_homography(p1n, p2n)
        best_H = torch.linalg.inv(T2) @ Hn @ T1

    return best_H, best_inliers


def homography_flow(H: torch.Tensor, Hsize: Tuple[int, int], Wsize: Tuple[int, int]) -> torch.Tensor:
    """
    Given H (3x3) and image size, return flow field from (x,y) -> (x',y')- (x,y)
    returns [1,2,H,W]
    """
    Ht, Wt = Hsize[0], Wsize[0]  # not used; we get them per-call
    raise NotImplementedError  # kept for clarity (unused)


def apply_homography_to_grid(H: torch.Tensor, H_img: int, W_img: int, device, dtype) -> torch.Tensor:
    """
    Map each pixel center via H.
    returns: flow_bg [1,2,H,W] in pixel units
    """
    xs = torch.arange(W_img, device=device, dtype=dtype)
    ys = torch.arange(H_img, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
    ones = torch.ones_like(xx)
    P = torch.stack([xx, yy, ones], dim=-1).reshape(-1, 3).T  # [3, H*W]
    Pp = (H @ P)  # [3, HW]
    Pp = Pp[:2] / (Pp[2:3] + 1e-8)  # [2, HW]
    disp = (Pp - P[:2]).reshape(2, H_img, W_img).unsqueeze(0)  # [1,2,H,W]
    return disp.float()


# ----------------------------
# 2D flow -> 3D unit direction (camera frame)
# ----------------------------

@torch.inference_mode()
def flow_to_3d_dir(
    flow: torch.Tensor,  # [1,2,H,W] (dx,dy) in pixels
    fx: float, fy: float, cx: float, cy: float,
    compensate_small: float = 0.5  # px threshold for zeroing
) -> torch.Tensor:
    """
    Map per-pixel flow to 3D direction in camera coords (unit), using tangent-sphere differential.
    Zeros small magnitudes.
    returns: [H,W,3]
    """
    _, _, H, W = flow.shape
    device = flow.device
    dtype = torch.float32

    # pixel grid
    xs = torch.arange(W, device=device, dtype=dtype)
    ys = torch.arange(H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
    x = (xx - cx) / fx
    y = (yy - cy) / fy

    du = flow[0, 0]
    dv = flow[0, 1]
    dx = du / fx
    dy = dv / fy

    denom = 1.0 + x * x + y * y
    # v = [dx, dy, 0] - [(x,y,1)] * ((x*dx + y*dy) / denom)
    dot = x * dx + y * dy
    vx = dx - x * (dot / denom)
    vy = dy - y * (dot / denom)
    vz = - (dot / denom)

    v = torch.stack([vx, vy, vz], dim=-1)  # [H,W,3]
    mag = torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1e-8)
    v = v / mag

    # zero out small flows
    small = (du.abs() + dv.abs()) * 0.5 < compensate_small
    v[small] = 0.0
    return v


# ----------------------------
# Main pipeline
# ----------------------------

class MotionFieldEstimator:
    def __init__(
        self,
        H: int,
        W: int,
        N: int = 4,                 # number of frames for accumulation
        levels: int = 2,
        search_radius: int = 3,
        patch_agg: int = 3,
        stride_for_ransac: int = 4, # subsample for correspondences
        intrinsics: Optional[Tuple[float, float, float, float]] = None,  # (fx,fy,cx,cy)
        ground_mode: bool = True,
        half_precision: bool = True,
        camera_compensated: bool = True
    ):
        self.H, self.W = H, W
        self.N = N
        self.levels = levels
        self.search_radius = search_radius
        self.patch_agg = patch_agg
        self.stride = stride_for_ransac
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype_work = torch.float16 if (half_precision and self.device.type == 'cuda') else torch.float32
        self.intrinsics = intrinsics or (0.9 * W, 0.9 * W, W * 0.5, H * 0.5)  # rough default fx=fy≈0.9*W
        self.ground_mode = ground_mode
        self.camera_compensated = camera_compensated

        self.frames = deque(maxlen=N)
        self.prev_gray = None
        self._ema_residual = None  # smooth over time

        torch.backends.cudnn.benchmark = True

    def _gray(self, frame_bgr_uint8: torch.Tensor) -> torch.Tensor:
        g = to_gray_torch(frame_bgr_uint8.to(self.device))
        if g.shape[-2:] != (self.H, self.W):
            g = F.interpolate(g, size=(self.H, self.W), mode='bilinear', align_corners=True)
        return g

    @torch.inference_mode()
    def update(self, frame_bgr_uint8: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Push a new frame and compute the vector field for the last step.
        frame_bgr_uint8: [H0,W0,3] uint8 tensor (on CPU ok)
        returns dict with:
          - vectors_3d: [H,W,3] float32 (camera coords), zeros for static
          - mask_moving: [H,W] bool
          - flow_raw, flow_bg, flow_res: [1,2,H,W]
          - H_bg: [3,3]
          - inliers_bg: [H,W] bool
        """
        g = self._gray(frame_bgr_uint8)
        out = {}
        if self.prev_gray is None:
            self.prev_gray = g
            # Initial empty outputs
            out['vectors_3d'] = torch.zeros((self.H, self.W, 3), device=self.device)
            out['mask_moving'] = torch.zeros((self.H, self.W), dtype=torch.bool, device=self.device)
            out['flow_raw'] = torch.zeros((1, 2, self.H, self.W), device=self.device)
            out['flow_bg'] = torch.zeros((1, 2, self.H, self.W), device=self.device)
            out['flow_res'] = torch.zeros((1, 2, self.H, self.W), device=self.device)
            out['H_bg'] = torch.eye(3, device=self.device)
            out['inliers_bg'] = torch.zeros((self.H, self.W), dtype=torch.bool, device=self.device)
            return out

        # 1) Dense optical flow (prev -> curr)
        flow_raw = dense_flow_correlate(
            self.prev_gray, g,
            levels=self.levels,
            search_radius=self.search_radius,
            patch_agg=self.patch_agg,
            dtype=self.dtype_work
        )  # [1,2,H,W]

        # 2) Background homography via RANSAC on a stride grid
        with torch.inference_mode():
            yy = torch.arange(self.H, device=self.device)
            xx = torch.arange(self.W, device=self.device)
            yy, xx = torch.meshgrid(yy, xx, indexing='ij')
            mask = ((yy % self.stride) == 0) & ((xx % self.stride) == 0)
            x0 = xx[mask].float()
            y0 = yy[mask].float()
            u = flow_raw[0, 0][mask]
            v = flow_raw[0, 1][mask]
            p1 = torch.stack([x0, y0], dim=1)  # [M,2]
            p2 = torch.stack([x0 + u, y0 + v], dim=1)
            H_bg, in_mask = fit_homography_ransac(p1, p2, iters=256, thresh=2.0)

        # 3) Background flow from homography, residual, moving mask
        flow_bg = apply_homography_to_grid(H_bg, self.H, self.W, self.device, torch.float32)  # [1,2,H,W]
        flow_res = flow_raw - flow_bg if self.camera_compensated else flow_raw
        mag_res = torch.sqrt(flow_res[:, 0] ** 2 + flow_res[:, 1] ** 2)[0]  # [H,W]

        # temporal smoothing (EMA)
        if self._ema_residual is None:
            ema = flow_res.clone()
        else:
            ema = 0.7 * self._ema_residual + 0.3 * flow_res
        self._ema_residual = ema

        # moving mask
        thresh_px = 0.75  # tune: pixels
        mask_moving = (torch.sqrt(ema[:, 0] ** 2 + ema[:, 1] ** 2)[0] > thresh_px)

        # 4) 2D -> 3D direction (camera frame), zero where not moving
        fx, fy, cx, cy = self.intrinsics
        vectors_3d_all = flow_to_3d_dir(
            ema, fx=fx, fy=fy, cx=cx, cy=cy, compensate_small=0.5
        )  # [H,W,3]

        vectors_3d = torch.zeros_like(vectors_3d_all)
        vectors_3d[mask_moving] = vectors_3d_all[mask_moving]

        # package outputs
        out['vectors_3d'] = vectors_3d
        out['mask_moving'] = mask_moving
        out['flow_raw'] = flow_raw
        out['flow_bg'] = flow_bg
        out['flow_res'] = flow_res
        out['H_bg'] = H_bg
        # expand inlier map back to full res for debug visualization
        in_full = torch.zeros((self.H, self.W), dtype=torch.bool, device=self.device)
        in_full[mask] = in_mask
        out['inliers_bg'] = in_full

        # update prev
        self.prev_gray = g
        return out


# ----------------------------
# Example: minimal webcam loop (optional)
# ----------------------------

def _maybe_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None

def demo_webcam(
    cam_index: int = 0,
    target_size: Tuple[int, int] = (480, 640),  # H,W (keep modest for FPS)
    N: int = 4
):
    cv2 = _maybe_cv2()
    if cv2 is None:
        print("OpenCV not installed; skipping demo.")
        return

    H, W = target_size
    est = MotionFieldEstimator(H, W, N=N, levels=2, search_radius=3, patch_agg=3,
                               stride_for_ransac=4, ground_mode=True, half_precision=True,
                               camera_compensated=True)

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_t = torch.from_numpy(frame[:, :, ::-1].copy())  # BGR->RGB not critical here
        out = est.update(frame_t)

        # Quick visualization: draw quiver on movers
        vis = frame.copy()
        yy, xx = torch.meshgrid(torch.arange(H, device=est.device),
                                torch.arange(W, device=est.device), indexing='ij')
        step = 12
        m = out['mask_moving'] & ((yy % step) == 0) & ((xx % step) == 0)
        pts = torch.stack([xx[m], yy[m]], dim=1).cpu().numpy()
        flow = out['flow_res'][0].permute(1,2,0)[m].cpu().numpy()
        for (x, y), (u, v) in zip(pts, flow):
            x2 = int(round(x + u))
            y2 = int(round(y + v))
            cv2.arrowedLine(vis, (int(x), int(y)), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

        cv2.imshow("motion vectors (camera-compensated)", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_webcam()
