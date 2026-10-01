"""Run inside a Colab notebook: browser downloads replace Google Drive persistence."""
import json
from pathlib import Path
import subprocess
import sys


def run_with_downloads(data="/content/chbmit-v2", output="/content/eeg-v2-run/full-seed42"):
    from google.colab import files
    from seizure_v2.recovery import PREFIX
    command = [sys.executable, "scripts/run_study.py", "--data", data,
               "--output", output, "--interactive-backups"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if not line.startswith(PREFIX):
                continue
            event = json.loads(line[len(PREFIX):])
            print(f"RECOVERY READY: {event['stage']} ({event['bytes'] / 1024**2:.2f} MiB)", flush=True)
            while True:
                answer = input("Type download, then enter the SHA-256 after saving and verifying locally: ").strip()
                if answer == "download":
                    files.download(event["path"])
                elif answer == event["sha256"]:
                    process.stdin.write(answer + "\n")
                    process.stdin.flush()
                    break
                else:
                    print("Not acknowledged. The experiment remains paused at its saved checkpoint.")
        if process.wait():
            raise RuntimeError("Study stopped; inspect the error and resume from the last verified recovery")
        release = Path(output) / "demo.tar.gz"
        from seizure_v2.common import sha256
        print(f"FINAL RELEASE: {release}; SHA256 {sha256(release)}")
        if input("Type download for the final demo release: ").strip() == "download":
            files.download(str(release))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


if __name__ == "__main__":
    run_with_downloads()
