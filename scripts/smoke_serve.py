"""Exercise the actual HTTP server, optionally in a serving-only environment."""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-lean", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if args.require_lean:
        for package in ["torch", "pyedflib", "sklearn", "matplotlib"]:
            assert importlib.util.find_spec(package) is None, f"Unexpected training/preparation dependency: {package}"
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "web.backend.app:app",
                                "--host", "127.0.0.1", "--port", str(args.port)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    def get(path):
        try:
            response = urlopen(f"http://127.0.0.1:{args.port}{path}", timeout=2)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.headers.get_content_type(), response.read()

    try:
        deadline = time.monotonic() + 20
        while True:
            if process.poll() is not None:
                raise RuntimeError(process.stderr.read())
            try:
                status, _, html = get("/")
                break
            except (URLError, TimeoutError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(.2)
        assert status == 200 and b"/static/style.css" in html
        for asset in ["style.css", "app.js", "plot.js"]:
            code, content_type, body = get("/static/" + asset)
            assert code == 200 and body and ("css" in content_type or "javascript" in content_type)
        status, _, body = get("/api/research")
        research = json.loads(body)
        assert status == 200 and research["status"] == "awaiting_artifacts" and not research["models"]
        assert get("/healthz")[0] == 503
        assert get("/api/examples")[0] == 503
        print("FastAPI HTTP smoke passed: HTML/assets/API served; missing model artifacts are explicitly unavailable.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
