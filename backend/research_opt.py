# -*- coding: utf-8 -*-
"""
研究脚本：分档适用范围 + 参数稳健性 + 置信区间
==============================================
A. 分档阈值(15/40/65)按类别 OOS 验证: 宽基阈值直接套用于行业/港股是否方向一致
B. 参数网格 IS/OOS 寻优: SMOOTH_W1(1年同比权重) / G_ADJUST_MAX(g调权幅度) / DCA_SPAN(定投斜率)
   目标函数 = 系数定投(B) 的 XIRR；仅当 IS 最优组合在 OOS 亦稳定且改善 >5% 才建议更换
C. OOS 分档未来3年收益的 bootstrap 置信区间(跨指数重采样)，检验分档区分度是否显著

共同方法: 严格无前视(每点只用截至当期数据)，前瞻3年收益，周频重叠样本。
局限: 存在幸存者偏差(仅现存指数)、无股息无交易成本、OOS 重叠样本有效量小，
      结论仅作方向参考，不作收益承诺。
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

os.environ.setdefault("DB_PATH", str(Path(__file__).resolve().parent.parent / "db" / "valuation.db"))

import numpy as np
import pandas as pd

import data_service as ds
import valuation as v
import backtest_dca as bt

OOS = "2022-01-01"
FWD = 3 * v.PTS_PER_YEAR
MIN_POINTS = 4 * v.PTS_PER_YEAR
RNG = np.random.default_rng(42)
BOOT = 2000

CATS = {"broad": "宽基", "strategy": "策略红利", "industry": "行业", "hk": "港股"}


def load_all():
    out = []
    for code, m in ds.INDICES.items():
        df = ds.load_series(code)
        if df is None or len(df) < MIN_POINTS:
            continue
        df = df.reset_index(drop=True)
        out.append((code, m, df, bt.align_y10(df)))
    return out


def fwd_returns(sig: pd.DataFrame) -> pd.DataFrame:
    """每个信号点 + 未来3年收益（不足3年的尾部点丢弃，避免部分区间）"""
    n = len(sig)
    close = sig["close"].to_numpy(dtype=float)
    rows = []
    for i in range(n - FWD):
        rows.append({"date": sig["date"].iloc[i], "composite": sig["composite"].iloc[i],
                     "fwd": close[i + FWD] / close[i] - 1})
    return pd.DataFrame(rows)


def bucket_of(c: float) -> str:
    if c < v.ZONE_LOW:
        return "<15%"
    if c < v.ZONE_MID:
        return "15-40%"
    if c < v.ZONE_HIGH:
        return "40-65%"
    return ">=65%"


BUCKETS = ["<15%", "15-40%", "40-65%", ">=65%"]


def part_a(data):
    print(f"\n{'=' * 88}\nA. 分档阈值按类别 OOS(>=2022) 未来3年收益验证\n{'-' * 88}")
    print(f"{'类别':<6}{'分档':>9}{'样本':>6}{'均值':>9}{'中位数':>9}{'胜率':>8}")
    print("-" * 88)
    # 逐类别聚合
    for cat, label in CATS.items():
        allr = []
        for code, m, df, y10 in data:
            if m["cat"] != cat:
                continue
            sig = bt.rolling_signals(df, y10, m["type"], m["cat"])
            r = fwd_returns(sig)
            r = r[r["date"] >= OOS]
            if len(r):
                r["bucket"] = r["composite"].map(bucket_of)
                allr.append(r)
        if not allr:
            continue
        r = pd.concat(allr)
        for b in BUCKETS:
            s = r[r["bucket"] == b]["fwd"]
            if len(s) == 0:
                print(f"{label:<6}{b:>9}{0:>6}{'--':>9}{'--':>9}{'--':>8}")
                continue
            print(f"{label:<6}{b:>9}{len(s):>6}{s.mean():>9.1%}{s.median():>9.1%}"
                  f"{(s > 0).mean():>8.0%}")
        print("-" * 88)


def run_b_xirr(sig: pd.DataFrame) -> float:
    return bt.run_strategy(sig, "dca")["xirr"]


def part_b(data):
    print(f"\n{'=' * 88}\nB. 参数网格 IS(<2022) 寻优 / OOS(>=2022) 验证 · 目标: 系数定投 XIRR 均值\n{'-' * 88}")
    grid = list(itertools.product([0.3, 0.4, 0.5], [0.08, 0.12, 0.16], [0.40, 0.50, 0.60]))
    rows = []
    for w1, gadj, span in grid:
        v.SMOOTH_W1, v.SMOOTH_W3 = w1, round(1 - w1, 2)
        v.G_ADJUST_MAX, v.DCA_SPAN = gadj, span
        is_r, oos_r = [], []
        for code, m, df, y10 in data:
            sig = bt.rolling_signals(df, y10, m["type"], m["cat"])
            d = pd.to_datetime(sig["date"])
            mi, mo = run_b_xirr(sig[d < OOS]), run_b_xirr(sig[d >= OOS])
            if not np.isnan(mi):
                is_r.append(mi)
            if not np.isnan(mo):
                oos_r.append(mo)
        rows.append({"w1": w1, "gadj": gadj, "span": span,
                     "IS": np.mean(is_r), "OOS": np.mean(oos_r)})
    df = pd.DataFrame(rows).sort_values("IS", ascending=False)
    print(f"{'W1':>5}{'G_ADJ':>7}{'SPAN':>6}{'IS 均值':>10}{'OOS 均值':>10}")
    for _, r in df.iterrows():
        star = "  <- 当前" if (r.w1, r.gadj, r.span) == (0.4, 0.12, 0.50) else ""
        print(f"{r.w1:>5.1f}{r.gadj:>7.2f}{r.span:>6.2f}{r['IS']:>10.2%}{r['OOS']:>10.2%}{star}")
    best = df.iloc[0]
    cur = df[(df.w1 == 0.4) & (df.gadj == 0.12) & (df.span == 0.50)].iloc[0]
    imp_is = best["IS"] - cur["IS"]
    imp_oos = best["OOS"] - cur["OOS"]
    print("-" * 88)
    print(f"IS 最优: W1={best.w1} G_ADJ={best.gadj} SPAN={best.span} "
          f"(IS {best['IS']:.2%} vs 当前 {cur['IS']:.2%}, OOS {best['OOS']:.2%} vs 当前 {cur['OOS']:.2%})")
    verdict = "维持当前参数" if (imp_is < 0.005 or best["OOS"] <= cur["OOS"]) else "建议评估更换"
    print(f"判据: IS 改善需 >0.5pp 且 OOS 不劣于当前 -> {verdict}"
          f"（IS 改善 {imp_is:+.2%}, OOS 改善 {imp_oos:+.2%}）")
    # 恢复默认
    v.SMOOTH_W1, v.SMOOTH_W3, v.G_ADJUST_MAX, v.DCA_SPAN = 0.4, 0.6, 0.12, 0.50


def part_c(data):
    print(f"\n{'=' * 88}\nC. OOS 分档未来3年收益 · bootstrap 95% CI（跨指数重采样 {BOOT} 次）\n{'-' * 88}")
    print(f"{'分档':>9}{'指数数':>7}{'均值':>9}{'95% CI':>18}{'正收益指数占比':>12}")
    print("-" * 88)
    for b in BUCKETS:
        means = []
        for code, m, df, y10 in data:
            sig = bt.rolling_signals(df, y10, m["type"], m["cat"])
            r = fwd_returns(sig)
            r = r[r["date"] >= OOS]
            if not len(r):
                continue
            s = r[r["composite"].map(bucket_of) == b]["fwd"]
            if len(s) >= 3:
                means.append(s.mean())
        means = np.array(means)
        if len(means) < 5:
            print(f"{b:>9}{len(means):>7}{'样本不足':>9}")
            continue
        boots = np.array([RNG.choice(means, len(means), replace=True).mean() for _ in range(BOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{b:>9}{len(means):>7}{means.mean():>9.1%}{f'[{lo:.1%}, {hi:.1%}]':>18}"
              f"{(means > 0).mean():>12.0%}")


def main():
    data = load_all()
    print(f"载入 {len(data)} 个指数（本地缓存，港股需另行拉取）")
    part_a(data)
    part_b(data)
    part_c(data)
    print(f"\n{'=' * 88}\n局限: 幸存者偏差(仅现存指数) / 周频重叠样本自相关 / 无股息与交易成本；"
          f"\n      bootstrap 为跨指数重采样，反映指数间离散度而非时间序列不确定性。")


if __name__ == "__main__":
    main()
