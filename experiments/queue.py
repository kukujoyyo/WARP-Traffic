"""Serial GPU job queue: runs each line of queue.txt once (lines may be appended while running)."""
import os, subprocess, time, sys

Q = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1] if len(sys.argv) > 1 else "queue.txt")
DONE = Q.replace(".txt", "_done.txt")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository folder; jobs run from here
LOG = os.path.join(os.path.dirname(ROOT), "logs")
PY = sys.executable   # start the queue with the Python of the project environment

while True:
    done = set(open(DONE).read().splitlines()) if os.path.exists(DONE) else set()
    todo = [l.strip() for l in open(Q).read().splitlines() if l.strip() and not l.startswith("#") and l.strip() not in done]
    if os.path.exists(Q.replace(".txt", ".pause")):   # pause between jobs; delete the file to continue
        time.sleep(30)
        continue
    if not todo:
        if os.path.exists(Q.replace(".txt", ".stop")):
            break
        time.sleep(30)
        continue
    cmd = todo[0]
    name = cmd.replace(" ", "_").replace("-", "").replace("/", "")[:120]
    t = time.time()
    for attempt in range(2):   # one retry (sporadic CUDA faults under WDDM)
        with open(os.path.join(LOG, f"q_{name}.log"), "a") as f:
            rc = subprocess.call(f'"{PY}" {cmd}', cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, shell=True)
        if rc == 0:
            break
    with open(DONE, "a") as f:
        f.write(cmd + "\n")
    with open(os.path.join(LOG, "queue_history.txt"), "a") as f:
        f.write(f"{time.strftime('%m-%d %H:%M')} rc={rc} {time.time() - t:.0f}s {cmd}\n")
