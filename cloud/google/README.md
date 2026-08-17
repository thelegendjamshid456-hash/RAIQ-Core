# RAIQ Google Training Launch Package

This directory prepares a future managed Google custom-training job for **RAIQ-200M v1**. It does **not** launch, purchase, provision, or submit training. Submission is prohibited until the repository production preflight passes with the approved corpus manifest and the target Google GPU environment.

> The primary worker-pool entry must have exactly one replica; additional workers belong in the second worker pool. Google’s managed training documentation describes this primary/worker ordering for distributed jobs.[1]

## Package contents

| File | Purpose |
|---|---|
| `Dockerfile` | GPU-enabled custom training container definition |
| `vertex_custom_job.template.yaml` | Placeholder custom-job worker-pool specification |
| `README.md` | Launch sequence and evidence gates |

## Required sequence before launch

First, replace all placeholder values in the template only after a licensed and approved production corpus exists. Build the container, push it to Artifact Registry, and upload the approved corpus/tokenizer artifacts to the selected Cloud Storage location. Then run the two-worker distributed smoke job, the checkpoint-resume smoke job, and the full production preflight **on the intended environment**. Only a preflight result of `production_pretraining_ready` permits a training submission.

A managed custom job can use a custom container and multiple worker pools; Google documents those two mechanisms and the corresponding job configuration fields.[1] [2] The RAIQ container must receive a cluster configuration that establishes rank, world size, local rank, and rendezvous information before calling the distributed training entry point. The current repository has local Gloo distributed primitives and must be extended and validated with NCCL on the selected GPU job before RAIQ-200M training begins.

## Do not submit this template yet

The template intentionally contains unresolved placeholders. It must not be passed to a submission command until the project owner has selected a region, project, Artifact Registry image, Cloud Storage output bucket, GPU machine type, GPU accelerator type/count, worker count, service account, VPC policy, and approved manifest version.

## References

[1]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/training/distributed-training "Google Cloud: Distributed training"
[2]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/training/create-custom-job "Google Cloud: Create a serverless training job"
