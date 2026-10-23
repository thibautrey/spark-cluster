# DGX Spark model discovery

The catalog defaults to **Best for DGX Spark**. Checkpoints explicitly tagged or named for
DGX Spark, GB10 or SM121 come first, followed by NVFP4 candidates. Cards display the signal;
the model sheet explains its evidence and runtime requirements. The other sort modes retain
Hugging Face's ordering while continuing to display these badges.

These are publisher-reported discovery signals, not runtime certifications or benchmark results.
NVFP4 metadata and exact tags are stronger signals than a repository name. An NVIDIA namespace
adds a small preference to an NVFP4 candidate; vendor identity alone earns no preference.
Generic Blackwell, FP4, FP8, AWQ, GPTQ, GGUF, MTP and base-model references do not establish
Spark optimization. Explicit datacenter-only targets receive no boost. A conflicting
quantization configuration suppresses an NVFP4 name or tag claim. GGUF candidates explain
their separate runtime requirement.

Spark ranking combines up to 48 trending matches, 48 NVFP4 repositories, 48 DGX Spark tagged
matches and 48 SM121 tagged matches. The NVFP4 supplement is restricted to the requested repo
or author substring; all requests retain task, library and access filters. The catalog reports
this bounded shortlist rather than claiming a global ordering of the entire Hub. Supplemental
source failures preserve regular results and appear in the status. Load more walks the same
ranked snapshot without duplicates; use another sort to browse beyond the shortlist.

The controller's existing SQLite cache stores the classified snapshot for ten minutes, keyed
by query, filters and classifier version. It survives restarts. Expired or mismatched pagination
cursors require restarting the search. Memory-fit results remain separately keyed by context,
concurrency and machine count. Optimization badges never bypass the fit or runtime readiness
checks, install code from a repository, or change the inference engine.

NVIDIA documents NVFP4's Spark workflow in the
[Model Optimizer playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/playbook-nvfp4-quantization).
Storage format and accelerated execution are separate: use the exact checkpoint with an
ARM64/SM121-compatible runtime and measure prefill, decode, quality, memory and communication
for your intended workload. SM100/DeepInfer targeting alone does not establish GB10 support.
