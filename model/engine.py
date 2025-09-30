# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

from model import seirdst
import util

# ---------- 形狀正規化工具 ----------

def _normalize_output_to_BHVI(output: torch.Tensor) -> torch.Tensor:
    """
    將模型輸出統一成 (B, H, V, 1)
    可能的輸入形狀：
      (B, H, V, 1)                → 直接回傳
      (B, 1, V, H)                → 轉成 (B, H, V, 1)
      (B, V, H) / (B, V, H, 1)    → 轉成 (B, H, V, 1)
      (B, T, V)                   → 視為 (B, H=T, V, 1)
    其他情形會丟錯，請在模型 forward 做調整。
    """
    if output.dim() == 4:
        B, A, C, D = output.shape
        # 已是 (B, H, V, 1)
        if D == 1 and A != 1:
            return output
        # (B, 1, V, H) → (B, H, V, 1)
        if A == 1:
            return output.permute(0, 3, 2, 1).contiguous()
        # (B, V, H, 1) → (B, H, V, 1)
        if C != 1 and D == 1:
            return output.permute(0, 2, 1, 3).contiguous()

    elif output.dim() == 3:
        # (B, V, H) 或 (B, T, V)
        B, A, C = output.shape
        # 多數情況 A=V、C=H → (B, V, H)；統一轉 (B, H, V, 1)
        # 若你的模型是 (B, T, V)，也會被轉成 (B, H=T, V, 1)
        return output.permute(0, 2, 1).unsqueeze(-1).contiguous()

    raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}")


def _normalize_real_to_BHVI(real_val: torch.Tensor) -> torch.Tensor:
    """
    將標籤 y 統一成 (B, H, V, 1)
    支援：
      (B, H, V, F) → 取病例 feature（第 0 維）→ (B, H, V, 1)
      (B, H, V)    → unsqueeze(-1)             → (B, H, V, 1)
      (B, V, H)    → permute → (B, H, V, 1)
    """
    if real_val.dim() == 4:
        # 取病例這個 feature（第 0 維）
        return real_val[..., 0:1].contiguous()

    if real_val.dim() == 3:
        B, A, C = real_val.shape
        # 猜 A 是 H、C 是 V；如果反了（A=V, C=H），自動轉
        if A <= 64 and C >= A:
            return real_val.unsqueeze(-1).contiguous()           # (B, H, V, 1)
        else:
            return real_val.permute(0, 2, 1).unsqueeze(-1).contiguous()  # (B, H, V, 1)

    raise ValueError(f"Unexpected real_val shape: {tuple(real_val.shape)}")


def _align_horizons(pred_bhvi: torch.Tensor, real_bhvi: torch.Tensor):
    """
    horizon 對齊：統一裁成相同 H（取較小者）
    兩者皆為 (B, H, V, 1)
    """
    Hp = pred_bhvi.size(1)
    Hr = real_bhvi.size(1)
    H  = min(Hp, Hr)
    return pred_bhvi[:, :H], real_bhvi[:, :H]


# ---------- 訓練器 ----------

class trainer:
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout,
                 lrate, wdecay, device, supports,
                 gat_bool=True, addaptadj=True, aptonly=False, noapt=False, aptinit=None):
        """
        與原本引數相容；模型輸出會在 train/eval 中被正規化成 (B, H, V, 1)。
        """
        self.model = seirdst(
            device=device,
            num_nodes=num_nodes,
            dropout=dropout,
            supports=supports,
            gat_bool=gat_bool,
            addaptadj=addaptadj,
            aptonly=aptonly,
            noapt=noapt,
            aptinit=aptinit,
            in_dim=in_dim,
            out_dim=1,                   # 預測病例
            residual_channels=nhid,
            dilation_channels=nhid,
            skip_channels=nhid * 8,
            end_channels=nhid * 16
        ).to(device)

        # receptive field 決定要 pad 幾步時間；沒有則視為 1
        rf = getattr(self.model, "receptive_field", 1)
        self.pad_t = max(0, int(rf) - 1)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input: torch.Tensor, real_val: torch.Tensor):
        self.model.train()
        self.optimizer.zero_grad()

        input = nn.functional.pad(input, (self.pad_t, 0, 0, 0))
        output, _ = self.model(input)                           # 模型輸出
        output_bhvi = _normalize_output_to_BHVI(output)         # (B, H, V, 1)
        real_bhvi   = _normalize_real_to_BHVI(real_val)         # (B, H, V, 1)
        output_bhvi, real_bhvi = _align_horizons(output_bhvi, real_bhvi)

        # === 重要：在標準化空間計算 loss（避免爆梯度） ===
        mean = torch.as_tensor(self.scaler.mean, device=output_bhvi.device, dtype=output_bhvi.dtype)
        std  = torch.as_tensor(self.scaler.std,  device=output_bhvi.device, dtype=output_bhvi.dtype)
        real_z = (real_bhvi - mean) / std
        # 不做遮罩（0 也參與訓練），用 L1 更穩；如要遮罩可仿造 util.masked_mae 的 mask
        loss = torch.mean(torch.abs(output_bhvi - real_z))
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # === 指標仍在原單位上回報（不影響梯度） ===
        predict_real = (output_bhvi * std) + mean
        mape = util.masked_mape(predict_real, real_bhvi, 0.0).item()
        rmse = util.masked_rmse(predict_real, real_bhvi, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input: torch.Tensor, real_val: torch.Tensor):
        self.model.eval()
        input = nn.functional.pad(input, (self.pad_t, 0, 0, 0))
        with torch.no_grad():
            output, _ = self.model(input)
            output_bhvi = _normalize_output_to_BHVI(output)
            real_bhvi   = _normalize_real_to_BHVI(real_val)
            output_bhvi, real_bhvi = _align_horizons(output_bhvi, real_bhvi)

            mean = torch.as_tensor(self.scaler.mean, device=output_bhvi.device, dtype=output_bhvi.dtype)
            std  = torch.as_tensor(self.scaler.std,  device=output_bhvi.device, dtype=output_bhvi.dtype)
            real_z = (real_bhvi - mean) / std
            val_loss = torch.mean(torch.abs(output_bhvi - real_z)).item()

            predict_real = (output_bhvi * std) + mean
            mape = util.masked_mape(predict_real, real_bhvi, 0.0).item()
            rmse = util.masked_rmse(predict_real, real_bhvi, 0.0).item()
        return val_loss, mape, rmse
