#!/usr/bin/env python3
"""RunPod lifecycle for training crop-health-dl on a rented GPU.

  pod.py up                     create/start the pod, wait for SSH
  pod.py status                 state, cost/hr, SSH endpoint
  pod.py push                   rsync code to /workspace/crop-health-dl
  pod.py run data               download PlantVillage onto the pod volume
  pod.py run smoke              two-batch sanity run
  pod.py run <config_name>      one config, e.g. resnet50_coarse
  pod.py run all                every config except smoke, in sequence
  pod.py submit <job> [job...]  start jobs detached on the pod and return;
                                the pod stops itself when they finish
  pod.py log                    tail the detached job's log
  pod.py collect [--weights]    start the pod if needed, pull results/, stop
  pod.py pull [--weights]       fetch results/ (model.keras only with --weights)
  pod.py shell                  interactive SSH
  pod.py down                   stop (billing stops, volume persists)
  pod.py destroy                terminate (y/N confirm)

`run` does up -> push -> bootstrap -> job -> pull -> down, and always stops
the pod, even on failure or Ctrl-C. Add --keep-up to leave it running.
--dry-run prints every request and command without executing anything.

RUNPOD_API_KEY comes from .env at the repo root, else the environment.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PODENV = Path(__file__).resolve().parent
ROOT = PODENV.parent
STATE_FILE = PODENV / ".pod_state.json"
SSH_KEY = PODENV / "id_ed25519"

API_BASE = "https://rest.runpod.io/v1"

POD_NAME = "crop-health-dl"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU_TYPE_IDS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA A40",
]
CONTAINER_DISK_GB = 20
VOLUME_GB = 30
REMOTE_DIR = "/workspace/crop-health-dl"
REMOTE_VENV = "/workspace/venv-tf"

WAIT_TIMEOUT_S = 900
POLL_INTERVAL_S = 5

SYNC_PATHS = ["src", "configs", "scripts", "data/download.py", "requirements.txt",
              "pyproject.toml", "Makefile", "podenv/bootstrap.sh"]


# --- local state and auth ---

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def clear_state():
    STATE_FILE.unlink(missing_ok=True)


def api_key_from_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if value:
                    return value
    return os.environ.get("RUNPOD_API_KEY")


def require_api_key(dry_run):
    if dry_run:
        return None
    key = api_key_from_env()
    if not key:
        sys.exit("RUNPOD_API_KEY not set (put it in .env or the environment)")
    return key


def ensure_ssh_key(dry_run):
    pub = SSH_KEY.with_suffix(".pub")
    if SSH_KEY.exists():
        return pub.read_text().strip()
    if dry_run:
        print(f"[dry-run] would generate {SSH_KEY}")
        return "ssh-ed25519 DRYRUN"
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-C", POD_NAME, "-f", str(SSH_KEY)],
        check=True, stdout=subprocess.DEVNULL,
    )
    return pub.read_text().strip()


# --- RunPod REST API ---

def api(method, path, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API_BASE}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 # Cloudflare in front of the API rejects urllib's default agent.
                 "User-Agent": f"{POD_NAME}/pod.py"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode()
            return resp.status, (json.loads(text) if text else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def pod_spec(cloud_type, public_key):
    return {
        "name": POD_NAME,
        "imageName": IMAGE,
        "gpuTypeIds": GPU_TYPE_IDS,
        "gpuCount": 1,
        "cloudType": cloud_type,
        "containerDiskInGb": CONTAINER_DISK_GB,
        "volumeInGb": VOLUME_GB,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        # Without this, Community Cloud may place the pod on a host that has
        # no public IP, and SSH never becomes reachable.
        "supportPublicIp": True,
        "minDownloadMbps": 200,
        "env": {"PUBLIC_KEY": public_key},
    }


def create_pod(key, public_key, dry_run):
    for cloud_type in ("COMMUNITY", "SECURE"):
        body = pod_spec(cloud_type, public_key)
        if dry_run:
            print(f"[dry-run] POST /pods cloudType={cloud_type}")
            print(json.dumps(body, indent=2))
            return {"id": "dryrun-pod"}
        status, payload = api("POST", "/pods", key, body)
        if status in (200, 201):
            return payload
        print(f"POST /pods ({cloud_type}) -> {status}: {payload}", file=sys.stderr)
    sys.exit("could not create a pod on either cloud type")


def get_pod(pod_id, key, dry_run):
    if dry_run:
        print(f"[dry-run] GET /pods/{pod_id}")
        return None
    status, payload = api("GET", f"/pods/{pod_id}", key)
    if status == 404:
        return None
    if status != 200:
        sys.exit(f"GET /pods/{pod_id} -> {status}: {payload}")
    return payload


def pod_action(action, pod_id, key, dry_run):
    if dry_run:
        print(f"[dry-run] {action} {pod_id}")
        return
    if action == "terminate":
        status, payload = api("DELETE", f"/pods/{pod_id}", key)
        ok = status in (200, 204)
    else:
        status, payload = api("POST", f"/pods/{pod_id}/{action}", key)
        ok = status == 200
    if not ok:
        raise RuntimeError(f"{action} {pod_id} -> {status}: {payload}")


def ensure_pod(key, dry_run):
    public_key = ensure_ssh_key(dry_run)
    pod_id = load_state().get("pod_id")
    pod = get_pod(pod_id, key, dry_run) if pod_id else None
    if pod_id and pod is None and not dry_run:
        print(f"saved pod {pod_id} no longer exists")
        pod_id = None

    if pod_id is None:
        pod = create_pod(key, public_key, dry_run)
        pod_id = pod["id"]
        if not dry_run:
            save_state({"pod_id": pod_id})
        print(f"pod {pod_id} created")
        return pod_id

    if (pod or {}).get("desiredStatus") == "RUNNING":
        print(f"pod {pod_id} already running")
        return pod_id

    print(f"starting pod {pod_id}")
    try:
        pod_action("start", pod_id, key, dry_run)
    except RuntimeError as e:
        # A stopped pod is pinned to its host; if the host is full it never
        # starts again. Terminate and recreate elsewhere.
        if "free GPUs" in str(e) or "resources" in str(e):
            print("host is full, recreating the pod on a new host")
            pod_action("terminate", pod_id, key, dry_run)
            clear_state()
            pod = create_pod(key, public_key, dry_run)
            pod_id = pod["id"]
            save_state({"pod_id": pod_id})
        else:
            sys.exit(str(e))
    return pod_id


# --- SSH ---

def ssh_args(port):
    return ["-i", str(SSH_KEY), "-p", str(port),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=10"]


def endpoint_of(pod):
    ip = pod.get("publicIp")
    port = (pod.get("portMappings") or {}).get("22")
    return (ip, int(port)) if ip and port else None


def ssh_ok(host, port):
    probe = subprocess.run(["ssh", *ssh_args(port), f"root@{host}", "true"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return probe.returncode == 0


def wait_for_ssh(pod_id, key, dry_run):
    if dry_run:
        return ("dryrun-host", 22)
    print(f"waiting for SSH on {pod_id}")
    deadline = time.time() + WAIT_TIMEOUT_S
    while time.time() < deadline:
        pod = get_pod(pod_id, key, dry_run)
        endpoint = endpoint_of(pod) if pod else None
        if endpoint and ssh_ok(*endpoint):
            print(f"ssh up: root@{endpoint[0]}:{endpoint[1]}")
            return endpoint
        time.sleep(POLL_INTERVAL_S)
    sys.exit(f"SSH not reachable within {WAIT_TIMEOUT_S}s")


def gpu_preflight(endpoint, pod_id, key, dry_run):
    if dry_run:
        return
    host, port = endpoint
    probe = subprocess.run(
        ["ssh", *ssh_args(port), f"root@{host}",
         "python3 -c \"import ctypes,sys;sys.exit(ctypes.CDLL('libcuda.so.1').cuInit(0))\""],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        print("GPU preflight ok")
        return
    print(f"broken GPU on this host (cuInit -> {probe.returncode}); terminating, run `up` again",
          file=sys.stderr)
    pod_action("terminate", pod_id, key, dry_run)
    clear_state()
    sys.exit(1)


def remote(endpoint, command, dry_run):
    host, port = endpoint
    cmd = ["ssh", *ssh_args(port), f"root@{host}", command]
    if dry_run:
        print(f"[dry-run] ssh root@{host} '{command}'")
        return 0
    return subprocess.call(cmd)


def rsync(sources, dest, port, dry_run, excludes=()):
    cmd = ["rsync", "-az", "--progress", "-e", "ssh " + " ".join(ssh_args(port))]
    for pattern in excludes:
        cmd += ["--exclude", pattern]
    cmd += [*sources, dest]
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return 0
    return subprocess.call(cmd)


def push(endpoint, dry_run):
    host, port = endpoint
    # The image ships without rsync and the container disk resets on stop.
    remote(endpoint,
           "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -qq -y rsync) "
           f">/dev/null; mkdir -p {REMOTE_DIR}/data {REMOTE_DIR}/podenv", dry_run)
    rc = 0
    for path in SYNC_PATHS:
        src = ROOT / path
        if not src.exists() and not dry_run:
            continue
        dest = f"root@{host}:{REMOTE_DIR}/{path}" if "/" in path else f"root@{host}:{REMOTE_DIR}/"
        rc |= rsync([str(src)], dest, port, dry_run, excludes=["__pycache__"])
    if rc:
        sys.exit("push failed")
    # Mendeley refuses downloads from datacenter IPs, so the prepared tfds
    # copy is uploaded from the laptop once; later pushes find it unchanged.
    dataset = ROOT / "data" / "tfds" / "plant_village"
    if dataset.exists() or dry_run:
        remote(endpoint, f"mkdir -p {REMOTE_DIR}/data/tfds", dry_run)
        if rsync([str(dataset)], f"root@{host}:{REMOTE_DIR}/data/tfds/", port, dry_run):
            sys.exit("dataset push failed")
    print(f"pushed to {REMOTE_DIR}")


def pull(endpoint, dry_run, weights=False):
    host, port = endpoint
    excludes = [] if weights else ["*.keras"]
    rc = rsync([f"root@{host}:{REMOTE_DIR}/results/"], str(ROOT / "results") + "/",
               port, dry_run, excludes=excludes)
    print("pulled results/" if rc == 0 else "pull failed (no results yet?)")


def connected(args):
    key = require_api_key(args.dry_run)
    if args.dry_run:
        return ("dryrun-pod", ("dryrun-host", 22), None)
    pod_id = load_state().get("pod_id")
    pod = get_pod(pod_id, key, False) if pod_id else None
    if not pod or pod.get("desiredStatus") != "RUNNING":
        sys.exit("pod is not running (run `pod.py up`)")
    endpoint = endpoint_of(pod)
    if not endpoint:
        sys.exit("pod has no SSH port yet, try again shortly")
    return pod_id, endpoint, key


# --- jobs ---

def config_names():
    return sorted(p.stem for p in (ROOT / "configs").glob("*.yaml"))


def job_command(job):
    activate = f"cd {REMOTE_DIR} && source {REMOTE_VENV}/bin/activate"
    if job == "data":
        return f"{activate} && python data/download.py"
    if job == "all":
        runs = " && ".join(f"python -m src.train --config configs/{c}.yaml --overwrite"
                           for c in config_names() if c != "smoke")
        return f"{activate} && {runs}"
    if job in config_names():
        return f"{activate} && python -m src.train --config configs/{job}.yaml --overwrite"
    sys.exit(f"unknown job {job!r}; use data, all, or one of {config_names()}")


# --- subcommands ---

def cmd_up(args):
    key = require_api_key(args.dry_run)
    pod_id = ensure_pod(key, args.dry_run)
    endpoint = wait_for_ssh(pod_id, key, args.dry_run)
    gpu_preflight(endpoint, pod_id, key, args.dry_run)
    print(f"ssh -i {SSH_KEY} -p {endpoint[1]} root@{endpoint[0]}")
    print("remember `pod.py down` when finished")


def cmd_status(args):
    key = require_api_key(False)
    pod_id = load_state().get("pod_id")
    if not pod_id:
        print("no pod yet (run `pod.py up`)")
        return
    pod = get_pod(pod_id, key, False)
    if pod is None:
        print(f"saved pod {pod_id} no longer exists")
        return
    gpu = (pod.get("machine") or {}).get("gpuTypeId") or pod.get("gpuTypeId")
    print(f"pod {pod_id}  status={pod.get('desiredStatus')}  gpu={gpu}  "
          f"${pod.get('costPerHr')}/hr")
    endpoint = endpoint_of(pod)
    if endpoint:
        print(f"ssh root@{endpoint[0]}:{endpoint[1]}  reachable={ssh_ok(*endpoint)}")


def cmd_push(args):
    _, endpoint, _ = connected(args)
    push(endpoint, args.dry_run)


def cmd_pull(args):
    _, endpoint, _ = connected(args)
    pull(endpoint, args.dry_run, args.weights)


def cmd_shell(args):
    _, (host, port), _ = connected(args)
    os.execvp("ssh", ["ssh", *ssh_args(port), f"root@{host}"])


def cmd_run(args):
    command = job_command(args.job)
    key = require_api_key(args.dry_run)
    pod_id = ensure_pod(key, args.dry_run)
    exit_code = 1
    terminated = False
    try:
        endpoint = wait_for_ssh(pod_id, key, args.dry_run)
        gpu_preflight(endpoint, pod_id, key, args.dry_run)
        push(endpoint, args.dry_run)
        rc = remote(endpoint, f"bash {REMOTE_DIR}/podenv/bootstrap.sh", args.dry_run)
        if rc == 42:
            print("broken GPU host, terminating; rerun to land elsewhere", file=sys.stderr)
            pod_action("terminate", pod_id, key, args.dry_run)
            clear_state()
            terminated = True
        elif rc != 0:
            print("bootstrap failed", file=sys.stderr)
        else:
            print(f"running: {args.job}")
            exit_code = remote(endpoint, command, args.dry_run)
            pull(endpoint, args.dry_run, args.weights)
    finally:
        if terminated or not STATE_FILE.exists():
            pass  # the pod was already terminated (broken host)
        elif args.keep_up:
            print("--keep-up: pod left running, billing continues")
        else:
            print("stopping pod")
            try:
                pod_action("stop", pod_id, key, args.dry_run)
            except Exception as e:
                print(f"failed to stop pod {pod_id}: {e}; check the RunPod console",
                      file=sys.stderr)
    sys.exit(exit_code)


JOB_LOG = f"{REMOTE_DIR}/job.log"
JOB_EXIT = f"{REMOTE_DIR}/job.exit"


def cmd_submit(args):
    commands = [job_command(job) for job in args.jobs]
    key = require_api_key(args.dry_run)
    pod_id = ensure_pod(key, args.dry_run)
    endpoint = wait_for_ssh(pod_id, key, args.dry_run)
    gpu_preflight(endpoint, pod_id, key, args.dry_run)
    push(endpoint, args.dry_run)
    rc = remote(endpoint, f"bash {REMOTE_DIR}/podenv/bootstrap.sh", args.dry_run)
    if rc != 0:
        pod_action("stop", pod_id, key, args.dry_run)
        sys.exit("bootstrap failed, pod stopped")
    # The job runs under nohup so it survives this SSH session ending, writes
    # its exit code, then stops its own pod through the REST API so a dead
    # laptop cannot leave it billing. (Community pods do not get RUNPOD_*
    # variables, so runpodctl cannot be used for this.)
    chain = " && ".join(f"({c})" for c in commands)
    stop = (f'curl -s -o /dev/null -X POST -A {POD_NAME}/pod.py '
            f'-H "Authorization: Bearer $RUNPOD_API_KEY" {API_BASE}/pods/{pod_id}/stop')
    script = f"rm -f {JOB_EXIT}; ({chain}); echo $? > {JOB_EXIT}; {stop}"
    launch = f"nohup bash -c '{script}' > {JOB_LOG} 2>&1 < /dev/null &"
    remote(endpoint, f"RUNPOD_API_KEY={key or 'dryrun'} {launch}", args.dry_run)
    print(f"submitted: {' && '.join(args.jobs)}")
    print("watch with `pod.py log`; when the pod shows EXITED run `pod.py collect`")


def cmd_log(args):
    _, endpoint, _ = connected(args)
    remote(endpoint, f"tail -n {args.lines} {JOB_LOG}", args.dry_run)


def cmd_collect(args):
    key = require_api_key(args.dry_run)
    pod_id = ensure_pod(key, args.dry_run)
    try:
        endpoint = wait_for_ssh(pod_id, key, args.dry_run)
        remote(endpoint, f"cat {JOB_EXIT} 2>/dev/null | sed 's/^/job exit code: /'", args.dry_run)
        rsync([f"root@{endpoint[0]}:{JOB_LOG}"], str(ROOT / "results" / "job.log"),
              endpoint[1], args.dry_run)
        pull(endpoint, args.dry_run, args.weights)
    finally:
        pod_action("stop", pod_id, key, args.dry_run)
        print("pod stopped")


def cmd_down(args):
    key = require_api_key(args.dry_run)
    pod_id = load_state().get("pod_id")
    if not pod_id:
        print("no pod to stop")
        return
    pod_action("stop", pod_id, key, args.dry_run)
    print(f"pod {pod_id} stopped, volume persists")


def cmd_destroy(args):
    key = require_api_key(args.dry_run)
    pod_id = load_state().get("pod_id")
    if not pod_id:
        print("no pod to destroy")
        return
    if not args.dry_run and input(f"terminate pod {pod_id} permanently? [y/N] ").lower() != "y":
        return
    pod_action("terminate", pod_id, key, args.dry_run)
    clear_state()
    print("terminated")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("up", "status", "push", "shell", "down", "destroy"):
        sub.add_parser(name)
    p = sub.add_parser("pull")
    p.add_argument("--weights", action="store_true", help="also fetch model.keras files")
    p = sub.add_parser("run")
    p.add_argument("job", help="data | smoke | all | <config name>")
    p.add_argument("--keep-up", action="store_true")
    p.add_argument("--weights", action="store_true", help="also fetch model.keras files")
    p = sub.add_parser("submit")
    p.add_argument("jobs", nargs="+", help="jobs to run in sequence, e.g. smoke all")
    p = sub.add_parser("log")
    p.add_argument("--lines", type=int, default=40)
    p = sub.add_parser("collect")
    p.add_argument("--weights", action="store_true", help="also fetch model.keras files")
    args = parser.parse_args()

    commands = {"up": cmd_up, "status": cmd_status, "push": cmd_push, "pull": cmd_pull,
                "shell": cmd_shell, "run": cmd_run, "down": cmd_down, "destroy": cmd_destroy,
                "submit": cmd_submit, "log": cmd_log, "collect": cmd_collect}
    commands[args.cmd](args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
