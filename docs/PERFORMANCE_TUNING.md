# Performance Tuning Guide

To maximize throughput and minimize latency on high-end hardware, apply the following optimizations.

## 1. Enable Hugepages (Critical for >100B Models)

When running massive models (e.g., Llama-3-405B, Grok-1) on CPU, TLB misses become a major bottleneck. We strongly recommend enabling Transparent Hugepages (THP):

```bash
# Enable THP
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

For production environments, allocating static 1GB hugepages at boot time offers the best stability.

## 2. NUMA Rebalancing Logs

DenseCore's **Intelligent MoE Offloading** prints logs when it migrates experts between NUMA nodes to maximize memory bandwidth ($u$).

**Log Example:**
```text
[INFO] ExpertProfiler: Hot experts detected: [4, 12, 55]
[INFO] NUMA Migration: Expert #4 (128MB) moved to Node 0 (Local) -> Latency -40%
```

- **Hot experts detected**: The scheduler identified these experts are frequently accessed.
- **NUMA Migration**: The engine physically moved the memory pages of Expert #4 to the CPU node performing the compute.
- **Latency -X%**: Estimated reduction in memory access latency.

If you see "Ping-Pong" warnings (experts moving back and forth rapidly), try increasing `moe_batch_strictness` in your scheduler config.
