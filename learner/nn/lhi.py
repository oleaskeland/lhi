import torch
import torch.nn as nn


class SmallFCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.net(x)



class LHI(nn.Module):
    """
    forward  =  Processor(+h)  ∘  Core(+h)  ∘  Processor(−h)
    Both maps are reversible & symplectic.
    """
    def __init__(self, h, shears, dim=2, hidden_dim=6):
        super().__init__()
        self.h = float(h)
        self.shears = int(shears)
        self.dim = dim

        # ── core nets (one pair per shear step) ───────────────────────────────
        self.v_nets = nn.ModuleList([SmallFCN(dim // 2, hidden_dim=hidden_dim)
                                     for _ in range(shears)])
        self.h_nets = nn.ModuleList([SmallFCN(dim // 2, hidden_dim=hidden_dim)
                                     for _ in range(shears)])

        # ── processor nets (single pair) ─────────────────────────────────────
        self.proc_v = SmallFCN(dim // 2, hidden_dim=hidden_dim)
        self.proc_h = SmallFCN(dim // 2, hidden_dim=hidden_dim)



    # ───────────────────────── basic shear helpers ───────────────────────────
    def _v_shear(self, p, q, fn, h_step):
        grad_q = fn(q)
        return p - grad_q * h_step, q

    def _h_shear(self, p, q, fn, h_step):
        grad_p = fn(p)
        return p, q + grad_p * h_step

    # ─────────────────────────── processor ─────────────────────────────
    def _processor(self, p, q, sign=1.0):
        """
        Symmetric V‑H‑V using the single processor nets.
        sign = +1  → preprocessing
        sign = −1  → postprocessing (= inverse of preprocessing)
        """
        h_half = 0.5 * sign * self.h
        h_full = sign * self.h

        p, q = self._v_shear(p, q, self.proc_v, h_half)
        p, q = self._h_shear(p, q, self.proc_h, h_full)
        p, q = self._v_shear(p, q, self.proc_v, h_half)
        return p, q

    # ─────────────────────────── core ──────────────────────────
    def _core(self, p, q):
        """
        Leapfrog with learned shear banks.
        """
        # per-shear base step
        h_i = self.h / self.shears


        # loop over the shear banks
        for i in range(self.shears):

            p, q = self._v_shear(p, q, self.v_nets[i], h_i * 1/2)  # ½ V
            p, q = self._h_shear(p, q, self.h_nets[i], h_i)  # H
            p, q = self._v_shear(p, q, self.v_nets[i], h_i * 1/2)  # ½ V

        return p, q

    # ───────────────────────────────── forward ───────────────────────────────
    def forward(self, x):
        """
        One full reversible time‑step.
        """
        z = x.requires_grad_(True)
        p, q = z.split(self.dim // 2, dim=-1)

        # processor (+h)
        p, q = self._processor(p, q, sign=1.0)

        # core (+h)
        p, q = self._core(p, q)

        # processor (−h)
        p, q = self._processor(p, q, sign=-1.0)

        return torch.cat([p, q], dim=-1)


    def predict(self, xh, steps=1, keepinitx=False, returnnp=False):
        if len(xh.size()) == 1:
            xh = xh.unsqueeze(0)

        dim = xh.size(-1)
        size = len(xh.size())

        if dim == self.dim:
            pred = [xh]
            for _ in range(steps):
                pred.append(self(pred[-1]))
        else:
            x0, h = xh[..., :-1], xh[..., -1:]
            pred = [x0]
            for _ in range(steps):
                pred.append(self(torch.cat([pred[-1], h], dim=-1)))

        if keepinitx:
            steps = steps + 1
        else:
            pred = pred[1:]

        res = torch.cat(pred, dim=-1).view([-1, steps, self.dim][2 - size:])

        if returnnp:
            numpy_res = res.cpu().detach().numpy()
            if len(xh.size()) == 2 and xh.size(0) == 1:
                return numpy_res.squeeze(0) if len(numpy_res.shape) > 2 else numpy_res
            return numpy_res

        return res
