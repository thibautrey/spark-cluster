# Machine inference admission and signed-body replay

Deployed source commits: `ab55306` (machine admission), `e856bb7` (body replay).

- Controller: `192.168.1.149:18771`, image `spark-cluster:machine-queue-e856bb7`,
  image ID `sha256:6017e0dbb52fc741131310a11e95f1b68aa8c603da07b71a1207276005a426cc`.
- Spark bridge: updated `model_pool.py` and `app.py` in its existing virtualenv;
  previous files retained alongside the deployment's bridge launcher.
- Previous controller retained as
  `spark-cluster-controller-before-machine-queue-20261102T123215Z`.
- Private controller descriptor retained under
  `/data/deployment-backups/machine-queue-e856bb7/controller.json`.

The controller no longer shares a 64-request admission counter across machines.
Each executing PoolNode enforces its own 64-request limit across all local models.

Signed node authentication consumed the request body before the response handler
attempted to read the raw ASGI receive channel again. The node inference route now
replays the authenticated body once and subsequently preserves disconnect handling.
Signature validation remains unchanged.

Live validation through the controller's `/v1/chat/completions` endpoint:

- Non-streaming: HTTP 200, answer `12`, 7.19 seconds including cold routing refresh.
- Streaming: HTTP 200, answer `12`, final `[DONE]`, 0.24 seconds.
- Controller healthy; bridge service active.
- Qwen runtime container `7b3972b72bff` unchanged and not restarted.

These are synthetic arithmetic prompts, with thinking disabled and 32 output tokens.
No model weights or private conversations were modified.
