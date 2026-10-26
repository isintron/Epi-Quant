"""Entry point for the Epi-Quant pywebview desktop UI.

Run with:
    python v2/webview/app.py
(from anywhere -- `api.py` adds `v2/` to `sys.path` itself, so this doesn't
need to run from inside `v2/`.)
"""

import os

import webview

from api import EpiQuantAPI

FRONTEND_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "index.html")
FRONTEND_URL = FRONTEND_INDEX + "#epi-quant-native"


def main() -> None:
    api = EpiQuantAPI()
    window = webview.create_window("Epi-Quant", FRONTEND_URL, js_api=api, width=1280, height=860, min_size=(960, 640))
    # Two-phase wiring: the window object doesn't exist yet when EpiQuantAPI
    # is constructed, but api.py's start_* methods need it to push progress
    # via window.evaluate_js.
    api.set_window(window)
    webview.start()


if __name__ == "__main__":
    main()
