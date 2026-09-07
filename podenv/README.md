# podenv

Rent one GPU on RunPod, train there, pull the results back, stop the pod.
Needs only `ssh` and `rsync` locally and `RUNPOD_API_KEY` in `.env`.

```bash
python3 podenv/pod.py run smoke            # first time: pod + venv + dataset + 2-batch check
python3 podenv/pod.py run resnet50_coarse  # one config
python3 podenv/pod.py run all              # all six configs, then stop
python3 podenv/pod.py status               # is anything still billing?
python3 podenv/pod.py down                 # stop; the 30 GB volume keeps venv + dataset
python3 podenv/pod.py destroy              # delete everything
```

`run` always stops the pod when it finishes or fails. Results land in
`results/<run>/` without the `model.keras` weights; pass `--weights` to fetch
them too.

The dataset is downloaded on the pod (fast link) rather than uploaded from the
laptop, and lives on the persistent volume with the TensorFlow venv, so only
the first run pays the setup time.
