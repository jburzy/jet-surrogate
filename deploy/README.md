# Deploying the reinterpretation service on CERN PaaS

One image (`Dockerfile`, pixi `infer` environment, released model in
`models/release/`) runs three roles: `jet-surrogate serve` (web pod),
`jet-surrogate worker` (worker pods) and `worker --cleanup-only` (CronJob).
Jobs live on a shared PersistentVolumeClaim as `jobs.db` (SQLite, WAL) plus
one directory per job. Configuration is the `ConfigMap` (`JS_*` variables).

## One-time setup (CERN account required)

1. paas.cern.ch: create the project `jet-surrogate`; request a quota of at
   least 4 CPU, 12 GiB memory, 50 GiB storage (2 workers + web + headroom).
2. webservices.web.cern.ch: register the site `jet-surrogate.web.cern.ch`,
   type PaaS, linked to the project, visibility internet (or CERN only).
   Decide on access: open with the upload and event limits, or CERN SSO.
3. Image: the GitHub workflow `build-image` publishes
   `ghcr.io/jburzy/jet-surrogate:latest` on every push to `main` touching
   the code or the model. Make the package public, or create a pull secret
   in the project and reference it from the deployments. Alternatively push
   the same image to CERN Harbor (`registry.cern.ch`) and change `images:` in
   `kustomization.yaml`.

## Deploy and operate

```bash
oc login --token=... --server=https://api.paas.okd.cern.ch:443
oc project jet-surrogate
oc apply -k deploy/paas
oc get pods; oc logs deploy/jet-surrogate-worker -f
oc rollout restart deploy/jet-surrogate-web deploy/jet-surrogate-worker   # after a new image
oc scale deploy/jet-surrogate-worker --replicas=4
```

Local dry run of the same image:

```bash
docker build -t jet-surrogate .
docker run --rm -p 8080:8080 -v $PWD/service_data:/data -e JS_SERVICE_DIR=/data jet-surrogate serve &
docker run --rm -v $PWD/service_data:/data -e JS_SERVICE_DIR=/data jet-surrogate worker
```

## Notes

- Uploads are deleted once processed; results and logs expire after
  `JS_JOB_TTL_HOURS` (CronJob).
- Throughput: ~7 events/s per worker core (HepMC parsing dominates), so a
  20k-event upload takes ~45 min on one worker. Add workers or lower
  `JS_MAX_EVENTS`; a compiled HepMC reader would give ~10x.
- Heavy inputs can be offloaded later to HTCondor (CERN) or to OSCER through
  a pull worker without changing the web layer.
- A new surrogate: replace `models/release/`, update `MODEL.md`, push, wait
  for `build-image`, `oc rollout restart`.
