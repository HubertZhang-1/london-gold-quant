# -*- coding: utf-8 -*-
"""Push alerts to a Feishu (飞书/Lark) group-robot webhook.

Used by the live monitor to alert on key events (trend reversal / it shorted / stop
triggered / crash). The webhook is a Feishu custom-bot URL:
   https://open.feishu.cn/open-apis/bot/v2/hook/<token>

Config (never commit the real secret — see .gitignore):
   Env var  FEISHU_WEBHOOK  -> full Feishu webhook URL
   A gitignored env file feishu_push.env (KEY=VALUE) is also read.

Usage:
   py scripts/feishu_push.py --msg "..."                 # send one message
   py scripts/feishu_push.py --msg "..." --dry-run        # print only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / "feishu_push.env"


def load_webhook():
    creds = {}
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            creds[key.strip()] = value.strip()
    if os.environ.get("FEISHU_WEBHOOK"):
        creds["FEISHU_WEBHOOK"] = os.environ["FEISHU_WEBHOOK"]
    return creds.get("FEISHU_WEBHOOK", "").strip()


def send_feishu(webhook: str, msg: str):
    import requests
    payload = {"msg_type": "text", "content": {"text": msg}}
    resp = requests.post(webhook, json=payload, timeout=15)
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    # Feishu returns code 0 on success.
    if body.get("code", 0) != 0 and body.get("StatusCode", 200) not in (0, 200):
        raise RuntimeError(f"飞书返回错误: {body}")
    return body


def main():
    parser = argparse.ArgumentParser(description="Send an alert to Feishu")
    parser.add_argument("--msg", default="⚠️ 黄金策略监控预警")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    webhook = load_webhook()
    if not webhook:
        print("[未配置飞书 webhook] 把地址放到 feishu_push.env: FEISHU_WEBHOOK=...")
        print(args.msg)
        return 2

    if args.dry_run:
        print("[干跑] " + args.msg)
        return 0

    try:
        body = send_feishu(webhook, args.msg)
    except Exception as exc:
        print(f"发送失败: {exc}")
        print(args.msg)
        return 1
    print(f"✅ 已发送到飞书 (code={body.get('code')}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
