# -*- coding: utf-8 -*-
"""Push the daily adaptive+circuit-breaker monitor decision to WeChat.

Primary channel: WeCom (企业微信) group-robot webhook. Optional fallback:
ServerChan (Server酱) SendKey / PushPlus token.

It formats the JSON snapshot written by ``scripts/monitor_bull_adaptive.py``
(``data/bull_adaptive_decision.json``) into a WeChat-friendly text message and
POSTs it. If no credential is configured it prints the message plus a how-to
instead of failing, so it is safe to run before you have a key.

Configuration (never commit real secrets — see .gitignore):
  * Env var   WECOM_WEBHOOK   -> full WeCom group-robot webhook URL
  * Env var   SERVERCHAN_KEY  -> ServerChan SendKey (prefix "SCT...", optional)
  * Env var   PUSHPLUS_TOKEN  -> PushPlus token (optional)
  * Env var   WECHAT_PROVIDER -> "wecom" (default) | "serverchan" | "pushplus"
  A gitignored env file ``wechat_push.env`` at the project root is also read
  (KEY=VALUE lines). Copy ``wechat_push.env.example`` and fill in your key.

Usage:
  py scripts/wechat_push.py                     # send (loads decision json + creds)
  py scripts/wechat_push.py --decision path.json
  py scripts/wechat_push.py --dry-run           # print the formatted message only
  py scripts/wechat_push.py --provider wecom
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Make sure emoji / CJK print to GBK consoles without crashing.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECISION_DEFAULT = PROJECT_ROOT / "data" / "bull_adaptive_decision.json"
ENV_FILE = PROJECT_ROOT / "wechat_push.env"

STATE_NAMES = {
    "BEAR": "🐻 熊市/无趋势 — 空仓",
    "EXTREME_VOL": "⚠️ 极端波动 — 空仓避险",
    "HIGH_VOL": "🌊 高波动 — 1x 低杠杆",
    "CLEAN_TREND": "🚀 干净趋势 — 10x 高杠杆",
    "BULL": "📈 普通牛市 — 5x 中杠杆",
    "CHOP": "⚖️ 震荡 — 2x 低杠杆",
}

STATE_NOTES = {
    "BEAR": "趋势未确立，建议空仓等待，等 bull 分>0.55 再参与。",
    "EXTREME_VOL": "波动率冲高，规避黑天鹅，空仓。",
    "HIGH_VOL": "波动偏大，只做 1x 低杠杆，谨慎。",
    "CLEAN_TREND": "干净上升趋势，可用 10x 高杠杆放大。",
    "BULL": "牛市确认，可用 5x 中杠杆。",
    "CHOP": "震荡市，只做 2x 低杠杆或观望。",
}

SIGNAL_NAMES = {0: "观望/空仓", 1: "做多", -1: "做空"}


def load_creds():
    """Merge gitignored env file + real environment (env wins)."""
    creds = {}
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            creds[key.strip()] = value.strip()
    # Real env overrides the file.
    for name in ("WECOM_WEBHOOK", "SERVERCHAN_KEY", "PUSHPLUS_TOKEN", "WECHAT_PROVIDER"):
        if os.environ.get(name):
            creds[name] = os.environ[name]
    return creds


def load_decision(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"决定文件不存在: {path}\n先运行: py scripts/monitor_bull_adaptive.py")
    # utf-8-sig tolerates a UTF-8 BOM sometimes written by Windows editors.
    with open(path, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    return data


def format_message(d: dict) -> str:
    """Build a friendly WeChat text message from the decision snapshot."""
    date = d.get("date", "?")
    close = d.get("close", float("nan"))
    state = d.get("state", "?")
    lev = d.get("suggested_leverage", float("nan"))
    risk = d.get("risk_pct", float("nan"))
    sig = int(d.get("signal", 0))
    micro = d.get("micro", float("nan"))

    state_line = STATE_NAMES.get(state, f"未知状态 ({state}) — 观望")
    action = SIGNAL_NAMES.get(sig, "观望/空仓")
    note = STATE_NOTES.get(state, "请人工复核市场状态。")

    price = f"${close:,.1f}" if close == close else "?"
    lev_txt = f"{lev:.0f}x" if lev == lev else "?"
    risk_txt = f"{risk * 100:.1f}%" if risk == risk else "?"
    micro_txt = f"{micro:+.3f}" if micro == micro else "?"

    lines = [
        "📊 伦敦金自适应策略 · 今日监控",
        "------------------------------",
        f"🗓 {date}    收盘 {price}",
        f"市场状态: {state_line}",
        f"建议杠杆: {lev_txt}   (单笔风险 {risk_txt})",
        f"信号方向: {action}   (micro={micro_txt})",
        "------------------------------",
    ]

    stop_dist = d.get("stop_dist", 0)
    tp_dist = d.get("tp_dist", 0)
    if sig != 0 and stop_dist and tp_dist:
        direction = 1 if sig > 0 else -1
        stop = close - direction * stop_dist
        target = close + direction * tp_dist
        lines += [
            f"🛑 建议止损: ${stop:,.1f}",
            f"🎯 建议目标: ${target:,.1f}",
            "------------------------------",
        ]

    lines.append(f"💡 {note}")
    return "\n".join(lines)


def send_wecom(webhook: str, msg: str):
    """POST a text message to a WeCom group-robot webhook."""
    import requests
    payload = {"msgtype": "text", "text": {"content": msg}}
    resp = requests.post(webhook, json=payload, timeout=15)
    body = resp.json()
    if body.get("errcode", 0) != 0:
        raise RuntimeError(f"企业微信返回错误: {body}")
    return body


def send_serverchan(key: str, msg: str):
    import requests
    resp = requests.post(
        f"https://sctapi.ftqq.com/{key}.send",
        data={"title": "伦敦金每日监控", "desp": msg.replace("\n", "\n\n")},
        timeout=15,
    )
    body = resp.json()
    if body.get("code", 0) != 0:
        raise RuntimeError(f"Server酱返回错误: {body}")
    return body


def send_pushplus(token: str, msg: str):
    import requests
    resp = requests.post(
        "https://www.pushplus.plus/send",
        json={"token": token, "title": "伦敦金每日监控", "content": msg},
        timeout=15,
    )
    body = resp.json()
    if body.get("code", 200) != 200:
        raise RuntimeError(f"PushPlus返回错误: {body}")
    return body


HOW_TO = """
【如何拿到企业微信群机器人 webhook】
1. 在手机微信中打开你常用的群（或新建一个群）。
2. 右上角「…」→「群机器人」→「添加机器人」→ 选「新创建一个机器人」。
3. 复制「Webhook 地址」。形如：
   https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxxxxx

【如何配置】
把复制到的地址粘贴到项目根目录的 wechat_push.env（gitignored，不会提交）：
   WECOM_WEBHOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxxxxx
也可直接用环境变量设置，env 优先级更高。

【验证发送】
   py scripts/wechat_push.py

【提示】部分企业微信 Webhook 要求加白名单(企业可信IP)；若报错请在企业微信管理后台添加。
""".strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decision", default=str(DECISION_DEFAULT))
    parser.add_argument("--provider", choices=["wecom", "serverchan", "pushplus"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="只打印消息，不发送")
    parser.add_argument("--do-not-send", action="store_true", help="别名兼容")
    args = parser.parse_args()

    decision = load_decision(Path(args.decision))
    msg = format_message(decision)

    creds = load_creds()
    provider = args.provider or creds.get("WECHAT_PROVIDER", "wecom")
    webhook = creds.get("WECOM_WEBHOOK", "").strip()
    serverchan_key = creds.get("SERVERCHAN_KEY", "").strip()
    pushplus_token = creds.get("PUSHPLUS_TOKEN", "").strip()

    # Determine whether a send path is actually available.
    available = (provider == "wecom" and webhook) or \
                (provider == "serverchan" and serverchan_key) or \
                (provider == "pushplus" and pushplus_token)

    if args.dry_run or args.do_not_send:
        print(msg)
        print("\n[干跑模式] 以上为将发送的消息（未实际发送）。")
        return 0

    if not available:
        # No key configured: show the message + how-to and exit non-zero so a
        # cron/script notices it needs configuring.
        print(msg)
        print("\n" + "=" * 60)
        print("未配置推送凭据。配置后即可自动发送到微信。")
        print(HOW_TO)
        return 2

    try:
        if provider == "wecom":
            send_wecom(webhook, msg)
        elif provider == "serverchan":
            send_serverchan(serverchan_key, msg)
        else:
            send_pushplus(pushplus_token, msg)
    except Exception as exc:  # network / http / json errors
        print(f"发送失败: {exc}")
        print("\n消息全文如下:\n")
        print(msg)
        return 1

    print(f"✅ 已通过 {provider} 发送到微信。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
