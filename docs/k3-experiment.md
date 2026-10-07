# K3: CPU-only partial memory experiment

**This is a source-derived state and parameter-gather envelope, not a full K3 training-memory prediction.** The actual K3 forward/backward graph was not executed. Activations, pipeline scheduling, EP token dispatch, and several workspace categories remain unmodeled. No fit/OOM verdict is emitted.

Source: [TorchTitan commit 948d65c](https://github.com/pytorch/torchtitan/tree/948d65c868c5fa8f0290bcf9e54b69f004721f54), particularly `models/kimi_k3/flavors.py`, `model.py`, `common/attention/kda.py`, `common/moe.py`, and `distributed/parallelism_context.py`.

## What ran

The experiment executes real `FakeTensorMode`, PyTorch functional fake all-gathers, `ExtendedMemTracker`, and the NCCL allocation ledger entirely on CPU. It manually schedules source-derived parameter shapes. It does **not** automatically instrument or import the full TorchTitan model.

The full flavor has 93 decoder layers (69 KDA, 24 MLA), width 7168, vocabulary 163840, 896 experts per MoE layer, top-16 routing, latent width 3584, and expert hidden width 3072. The inventory includes every expert, the dense first layer, shared experts, embedding and output head, and the vision encoder. Source-shape arithmetic gives **2,779,931,738,208 parameters**; this is not verified by importing/constructing the actual model.

Parameter arithmetic uses every weight shape in `flavors.py`: a linear contributes input × output elements; the routed experts contribute `92 × 896 × 3 × 3584 × 3072`; depthwise KDA convolutions contribute channels × kernel width; RMSNorm contributes one vector. KDA also contributes `A_log` and `dt_bias`; the vision encoder contributes its learned spatial position table. `inventory()` exposes a per-layer dense/expert breakdown. Router quantile histograms and count/bias buffers are accounted separately from parameters. Rotary/cache buffers and exact FSDP padding are excluded.

## Illustrative configuration and results

The configuration is **not a user hardware prescription**: PP=8, dense DP shard=64, EP=8, TP=CP=DP replicate=1, 512 ranks. EP overlaps DP: expert-DP=64/8=8, so expert parameters are divided by EP8 and EDP8, **not DP64 and EP8**. Layers are assigned contiguously with nearly equal layer counts; this is **not the pinned TorchTitan pipeline schedule**.

State consists of FP32 parameters, FP32 gradients, and two FP32 Adam moments: 16 bytes per parameter, with no additional master copy. These are modeled co-resident. Gathered parameters are BF16, with two layer outputs retained concurrently and cast-input staging included. Batch size, sequence length and checkpointing are unset because activation lifetimes are not modeled.

| Case | Worst-stage partial envelope |
|---|---:|
| NCCL omitted | 101.372 GiB |
| Source-derived shared NET component, 16 channels, 3 assumed communicators | 101.560 GiB |
| Explicit allowance of 128 MiB × 3 assumed communicators | 101.747 GiB |

Maximum-stage static FP32 training state alone is 84.310 GiB. These totals exclude the unmodeled terms above. A zero NCCL allowance means **omitted**, not zero actual NCCL memory. The 128 MiB figure is **an uncalibrated assumption**, not a prediction. The NET profile covers only send/receive shared GDR P2P pools under declared assumptions (64 MiB per communicator here); it excludes other transport/plugin/context allocations and cannot be interpreted as total NCCL memory. Each JSON preserves profile provenance and exclusions.

Reproduce from the repository root after installation:

```bash
python -m memtracker_nccl.k3_experiment --output experiments/k3_512rank_partial.json
python -m memtracker_nccl.k3_experiment --nccl-mib-per-communicator 0 --output experiments/k3_512rank_no_nccl.json
python -m memtracker_nccl.k3_experiment --nccl-profile shared-net-component --channels 16 --output experiments/k3_512rank_shared_net_component.json
```

Override `--pp`, `--dp`, `--ep`, `--prefetch`, and explicit `--activation-mib` allowance. An activation allowance is an input assumption; the program does not estimate it.

## Exact-source function probe

An independent CPU experiment extracts the **unchanged body** of `_apply_attention_residual` from the pinned `model.py`, verifies the checkout and file against Git, and executes fake forward **and backward**. `nn.Linear` and `nn.RMSNorm` supply only the weight/epsilon attributes used by the function. No custom attention or MoE implementation is replaced or implied to run.

At 8192 tokens, width 7168, and 8 existing residual entries, the helper's simulated live-tensor peak is **13,858,037,764 bytes (12.906 GiB)**. This narrow eager result must not be added blindly to the envelope above: full-model scheduling, checkpointing and compiler fusion determine overlap and lifetimes.

```bash
git clone https://github.com/pytorch/torchtitan.git ../torchtitan
git -C ../torchtitan checkout 948d65c868c5fa8f0290bcf9e54b69f004721f54
python -m memtracker_nccl.k3_source_probe ../torchtitan --output experiments/k3_attention_residual_probe.json
```

The source `model.py` SHA256 is `37f27337dc7b17de3dee5c3a64fbcd797c58fa23767248627a717bd1716e0e13`.

## Full-model execution blocker and next integration work

On the tested macOS CPU host, importing the pinned model fails in `attn_gym.linear.kda.fwd.triton.l2norm_fwd` with `ModuleNotFoundError: No module named 'triton'`, after installing pinned `spmd_types==0.2.5`, `attn-gym==0.0.16`, and TorchTitan's pinned `torch_remat`. Triton has no compatible standard wheel for this host. This is a host/dependency blocker, not proof that all Linux CPU fake execution is impossible.

On a suitable Linux environment, install the pinned TorchTitan dependencies and attempt actual metadata-only construction and fake forward/backward. KDA, grouped experts, data-dependent routing, and GPU capability checks still need validation; appropriate fake implementations must model saved tensors and scratch memory, not only output shapes. Next connect actual FSDP/EP/PP fake collective sites and their waits to the ledger, then calibrate NCCL profiles against intended topology/version. The current prototype does not promise this integration already works.

Tested with PyTorch 2.14.1, Python 3.12, without CUDA. JSON reports preserve byte counts, per-stage accounting, source commit, scenario assumptions, and explicit missing coverage.
