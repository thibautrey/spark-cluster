# Stored models and space recovery

**Models → Downloaded models → Stored on disk** shows actual files, independently of download history or whether
a model is loaded in GPU memory. Filter by node to see its accessible model disks. Shared NFS
model storage appears once and identifies the nodes that share it; deleting that copy affects
all of those nodes. A controller copy and a remote cluster copy are separate deletion targets.

The inventory scans the configured model directory and Hugging Face cache. It includes managed
weights, recognized unmanaged weight folders/GGUF files, and cached Hugging Face repositories.
Known recipe runtime images and compilation/PLE cache volumes appear in a separate Docker group.
Unrelated Docker images/volumes are never offered for cleanup. Host directories that are not
mounted into the controller/bridge are outside this inventory; connect their storage first.

For each stored item the UI shows its location, physical size, estimated reclaimable bytes,
recipe references, and any deletion blocker. Disk free/total values come from the actual mounted
filesystem. Hard-linked files outside the selected directory are excluded from reclaim estimates;
symlink targets are not counted or followed. Docker image estimates exclude shared layers.
Docker does not report host free capacity through its standard API, so that value remains unknown
and is never added to a model disk's free space.

## Deleting a copy

Choose **Delete files…**, inspect the exact location and recipe references, select files, then
choose **Delete selected files permanently**. This removes only the chosen storage copy. It does
not unload runtimes, remove copies on other nodes, delete recipe definitions/version history,
or prune unrelated Docker resources. Re-downloads may be needed before affected recipes run again.

Deletion validates fresh server-side identifiers and checks active downloads, recipe installations,
managed runtime residency, and Docker container references. Files used by running models cannot be
selected. Unload the model or stop its transfer first. Docker images/cache volumes referenced by
containers are protected, and Docker deletion never uses `force`. If Docker usage cannot be read,
file deletion is blocked rather than guessing that the files are idle.

Filesystem deletion and new local model loads/download admission share a lock. The selected batch
is validated before deleting any item on that physical node. Deletion across several nodes is not
transactional: the UI reports individual failures and refreshes the inventory after successful
removals. File changes since selection invalidate the old identifier and require a refresh.

## Space errors and retry

Model downloads and recipe installation check the target's filesystem before accepting the work.
Remote delivery checks both the controller and cluster storage, because deleting a controller copy
cannot make space on an unrelated Spark disk. Existing exact weights are reused. Model startup
also requires 64 MiB of disk headroom for metadata and small cache files; weights already stored on
disk are not counted as another full download.

A rejected operation opens **Free disk space to continue**, showing the affected disk, bytes needed,
free bytes, and the remaining shortfall. Select other models or recipe artifacts to remove. The
operation's own weights are excluded from deletion. Free space is measured again after cleanup;
only an explicit **Retry operation** resumes the rejected request. Closing the chooser leaves it
rejected. Successful asynchronous operations are never retried automatically.

A Docker-root shortage may only become known while pulling an image or creating a runtime, because
Docker does not expose free host disk capacity. Such failures remain explicit disk-space errors.
Failed recipe/launch jobs offer **Free disk space and retry** to inspect node storage before a new
attempt. Docker artifacts may reside on a different disk from model weights, so reclaiming one does
not imply that the other gained capacity. A sufficiently large runtime/compiler cache still needs
headroom beyond the small startup reserve.

All storage endpoints require the configured model administration token when set. Deletion also
requires the token to be configured. Bridges must run this version for remote inventory/deletion;
an older/unreachable bridge shows an availability error rather than a false empty inventory.

HTTP surfaces:

- `GET /api/v1/storage`: controller and registered pools, with node membership and scoped asset IDs.
- `GET /api/v1/storage/node`: physical bridge/controller storage only.
- `POST /api/v1/storage/delete`: selected controller/cluster-qualified asset IDs.
- `POST /api/v1/storage/node/delete`: authenticated physical-node deletion.
- `POST /api/v1/storage/preflight`: check a model download without queuing it.
- Insufficient filesystem capacity returns HTTP **507** with `detail.code=insufficient_storage`
  and per-location requirements for the recovery chooser.
