"""
live_view.py - live per-step visualization for the curious-maze agent.

Shows, updating every step of every episode:
  - the agent's actual observation        (RGB + depth)
  - the agent's prediction of next obs     (RGB + depth)  -> step_dict['pred_obs_q']
  - the action it took                     (text)

Observations and predictions are RGB-D tensors keyed by observation name,
e.g. {'see_image': tensor of shape (1, 1, H, W, 4)} (channels-last).
Pure matplotlib in interactive mode: no Qt, no threading. Opens its own
window and coexists fine with PyBullet's GUI window.

Quick self-test (no agent / pybullet needed):
    python live_view.py
"""

from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Backend: a live, separate window needs an *interactive* matplotlib backend.
# ----------------------------------------------------------------------------
_INTERACTIVE = {
    "qtagg", "qt5agg", "qt4agg", "tkagg", "macosx",
    "gtk3agg", "gtk4agg", "wxagg", "nbagg", "webagg",
}


def _ensure_interactive_backend(verbose=True):
    if matplotlib.get_backend().lower() in _INTERACTIVE:
        return matplotlib.get_backend()
    for cand in ("QtAgg", "Qt5Agg", "TkAgg"):
        try:
            plt.switch_backend(cand)
            if verbose:
                print(f"[live_view] switched matplotlib backend to {cand}")
            return cand
        except Exception:
            continue
    if verbose:
        print(
            f"[live_view] WARNING: backend {matplotlib.get_backend()!r} may not "
            "show a live window.\n"
            "  - In a VS Code Interactive / Jupyter window, run:  %matplotlib qt\n"
            "  - Or put  matplotlib.use('QtAgg')  at the very TOP of main.py,\n"
            "    before 'import matplotlib.pyplot as plt'."
        )
    return matplotlib.get_backend()


# ----------------------------------------------------------------------------
# Array helpers (torch imported lazily -> no hard torch dependency here).
# ----------------------------------------------------------------------------
def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu").float().numpy()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


def to_hwc(x):
    """Convert an array/tensor to (H, W, C), channels-last.
    Handles batched (1, 1, H, W, 4) and channels-first (C, H, W)."""
    a = np.squeeze(_to_numpy(x))
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim == 2:
        return a[..., None]
    if a.ndim > 3:
        a = a.reshape((-1,) + a.shape[-3:])[0]
    if a.shape[-1] in (1, 3, 4):
        return a
    if a.shape[0] in (1, 3, 4):
        return np.transpose(a, (1, 2, 0))
    return a


def split_rgb_depth(hwc):
    """Split (H, W, C) into (rgb, depth); either may be None."""
    c = hwc.shape[-1]
    rgb = np.clip(hwc[..., :3], 0.0, 1.0) if c >= 3 else None
    if c == 4:
        depth = hwc[..., 3]
    elif c == 1:
        depth = hwc[..., 0]
    else:
        depth = None
    return rgb, depth


def describe(obj, _depth=0, _max_depth=4):
    """Print the nested structure (keys + tensor shapes) of e.g. step_dict.
    Handy if you change the agent and want to re-find a field."""
    import numbers
    pad = "  " * _depth
    if _depth > _max_depth:
        print(pad + "..."); return
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{pad}{k}:"); describe(v, _depth + 1, _max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{type(obj).__name__}[{len(obj)}]")
        if len(obj):
            describe(obj[0], _depth + 1, _max_depth)
    else:
        shape = getattr(obj, "shape", None)
        if shape is not None:
            print(f"{pad}<{type(obj).__name__} shape={tuple(shape)}>")
        elif isinstance(obj, numbers.Number):
            print(f"{pad}{obj}")
        else:
            print(f"{pad}<{type(obj).__name__}>")


# ----------------------------------------------------------------------------
# The live window.
# ----------------------------------------------------------------------------
class LiveView:
    def __init__(self, image_size=8, obs_key=None, depth_cmap="gray",
                 pause=0.001, title="Maze agent - live view"):
        _ensure_interactive_backend()
        self.image_size = image_size
        self.obs_key = obs_key          # which observation to show; None -> first key
        self.pause = pause
        self.depth_cmap = depth_cmap

        plt.ion()
        self.fig = plt.figure(figsize=(6.2, 6.6))
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

        gs = self.fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.30],
                                   hspace=0.35, wspace=0.20)
        self.ax = {
            "arr": self.fig.add_subplot(gs[0, 0]),   # actual rgb
            "prr_p": self.fig.add_subplot(gs[0, 1]),   # predicted rgb_p
            "prr_q": self.fig.add_subplot(gs[0, 2]),   # predicted rgb_q
            "ard": self.fig.add_subplot(gs[1, 0]),   # actual depth
            "prd_p": self.fig.add_subplot(gs[1, 1]),   # predicted depth_p
            "prd_q": self.fig.add_subplot(gs[1, 2]),   # predicted depth_q
        }
        titles = {
            "arr": "Actual RGB",     "prr_p": "Predicted RGB (Prior)",          "prr_q": "Predicted RGB (Post)",
            "ard": "Actual depth",   "prd_p": "Predicted depth (Prior)",    "prd_q": "Predicted dept (Post)",
        }
        for k, axx in self.ax.items():
            axx.set_title(titles[k], fontsize=10)
            axx.set_xticks([]); axx.set_yticks([]) 

        self.text_ax = self.fig.add_subplot(gs[2, :])
        self.text_ax.axis("off")
        self._text = self.text_ax.text(
            0.0, 0.9, "", va="top", ha="left", fontsize=11,
            family="monospace", transform=self.text_ax.transAxes)

        self._im = {}
        self.fig.show()

    def _pick(self, x):
        """If x is a {obs_name: tensor} dict, pull the chosen observation."""
        if isinstance(x, dict):
            if not x:
                return None
            key = self.obs_key if self.obs_key in x else sorted(x.keys())[0]
            return x[key]
        return x

    def _draw_one(self, key, data, is_rgb):
        if data is None:
            data = (np.zeros((self.image_size, self.image_size, 3), np.float32)
                    if is_rgb else np.zeros((self.image_size, self.image_size), np.float32))
        if key not in self._im:
            if is_rgb:
                self._im[key] = self.ax[key].imshow(data, interpolation="nearest")
            else:
                self._im[key] = self.ax[key].imshow(
                    data, interpolation="nearest", cmap=self.depth_cmap,
                    vmin=0.0, vmax=1.0)
        else:
            self._im[key].set_data(data)

    def update(self, real_image, pred_image_p, pred_image_q, action_text="", extra_text=""):
        """Call once per step. actual_obs / predicted_obs may each be a tensor
        or a {obs_name: tensor} dict (e.g. obs and step_dict['pred_obs_q'])."""
        a_rgb, a_d = real_image[:,:,:-1], real_image[:,:,-1]
        p_rgb, p_d = pred_image_p[:,:,:-1], pred_image_p[:,:,-1]
        q_rgb, q_d = pred_image_q[:,:,:-1], pred_image_q[:,:,-1]

        self._draw_one("arr", a_rgb, True)
        self._draw_one("prr_p", p_rgb, True)
        self._draw_one("prr_q", q_rgb, True)
        # Both actual and predicted obs live in [0,1] (the decoder ends with
        # (tanh + 1) / 2), so the two depth panels share one fixed 0..1 scale
        # and can be compared pixel-for-pixel.
        self._draw_one("ard", a_d, False)   # actual depth:    0..1
        self._draw_one("prd_p", p_d, False)   # predicted depth: 0..1
        self._draw_one("prd_q", q_d, False)   # predicted depth: 0..1

        msg = action_text or ""
        if extra_text:
            msg = (msg + "\n" + extra_text) if msg else extra_text
        self._text.set_text(msg)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        if self.pause:
            plt.pause(self.pause)

    def close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Standalone self-test with random data (no agent / pybullet needed).
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    view = LiveView(image_size=8)
    for step in range(80):
        actual = {"see_image": np.random.rand(1, 1, 8, 8, 4).astype("float32")}
        pred = {"see_image": np.random.rand(1, 1, 8, 8, 4).astype("float32")}
        view.update(
            actual, pred,
            action_text=f"Yaw: {np.random.randint(-90, 91)}. "
                        f"Speed: {np.random.randint(0, 76)}.",
            extra_text=f"demo step {step}")
        time.sleep(0.05)
    print("demo done - close the window to exit")
    plt.ioff(); plt.show()