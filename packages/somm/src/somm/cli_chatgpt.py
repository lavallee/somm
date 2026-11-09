"""Explicit, Somm-owned ChatGPT plan authorization and quota inspection."""
from __future__ import annotations

import json
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

from somm.chatgpt_auth import PERMISSION, ChatGPTAuth, ChatGPTAuthError, read_json


def command(args) -> int:
    auth = ChatGPTAuth()
    try:
        if args.chatgpt_cmd == "quota":
            from dataclasses import asdict

            from somm.codex_account import WeeklyQuotaProbe

            value = WeeklyQuotaProbe().read()
            print(json.dumps(asdict(value)))
            return 0 if value.used_percent is not None else 1
        if args.chatgpt_cmd == "models":
            from somm.providers.chatgpt import ChatGPTProvider

            print(json.dumps([m.name for m in ChatGPTProvider(auth=auth).models()]))
            return 0
        if args.chatgpt_cmd == "status":
            value = read_json(auth.path)
            print(json.dumps({"connected": PERMISSION in value.get("scopes", []),
                              "expires_at": value.get("expires_at")}))
            return 0
        callback = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if urlsplit(self.path).path != "/auth/callback":
                    self.send_error(404)
                    return
                query = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                if query.get("state") != pending["state"]:
                    self.send_error(400)
                    return
                callback.update(query)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Return to Somm to check the connection.")

            def log_message(self, *_args):
                pass  # OAuth callback URLs contain single-use secrets.

        with HTTPServer(("127.0.0.1", args.port), Handler) as server:
            server.timeout = 1
            url, pending = auth.begin(f"http://127.0.0.1:{server.server_port}/auth/callback")
            print("Open this on the computer running Somm login (or forward its callback port):", flush=True)
            print(url, flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            deadline = time.monotonic() + 600
            while not callback and time.monotonic() < deadline:
                server.handle_request()
            if not callback:
                raise ChatGPTAuthError("Sign-in timed out; no credentials were activated")
        result = auth.finish(pending, callback)
        print(json.dumps(result))
        return 0 if result["connected"] else 1
    except ChatGPTAuthError as exc:
        print(str(exc))
        return 1


def add_parser(sub):
    group = sub.add_parser("chatgpt", help="Somm-owned ChatGPT subscription connection")
    commands = group.add_subparsers(dest="chatgpt_cmd", required=True)
    login = commands.add_parser("login")
    login.add_argument("--port", type=int, default=1456)
    login.add_argument("--no-browser", action="store_true")
    login.set_defaults(func=command)
    for name in ("status", "models", "quota"):
        commands.add_parser(name).set_defaults(func=command)
