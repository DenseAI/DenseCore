void cb_moe_forward(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                    int nth, void* userdata) {
    (void)nth;
    const auto profile_begin =
        (ith == 0 && IsQwen36ProfilingEnabled()) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (ith == 0) {
        g_moe_callback_entry_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (IsMoEDebugLoggingEnabled() && ith == 0) {
        static std::atomic<int> moe_call_count{0};
        int call_id = moe_call_count.fetch_add(1);
        fprintf(stderr, "[DBG] cb_moe_forward #%d src0=[%lld,%lld] src1=[%lld,%lld]\n", call_id, (long long)src0->ne[0],
                (long long)src0->ne[1], (long long)src1->ne[0], (long long)src1->ne[1]);
        if (call_id < 8) {
            DebugLogTensorFiniteStats("MOE_IN", src0);
            DebugLogTensorFiniteStats("MOE_GATE", src1);
        }
    }
    if (ith != 0) {
        return;
    }
    auto* ud = static_cast<MoEUserData*>(userdata);
    if (!ud || !ud->layer) {
        FailClosedMoECallback(dst, &g_moe_callback_missing_userdata_count, "MoE callback missing userdata/layer");
        return;
    }
    if (!ud->backend) {
        // Fail-safe path for environments where registry CPU backend resolution is delayed
        // or unavailable at graph wiring time. Keep MoE execution on CPU rather than silently
        // dropping routed expert forward.
        ud->backend = &densecore::GetTelemetryCpuBackend();
    }
    if (!ud->backend) {
        FailClosedMoECallback(dst, &g_moe_callback_missing_backend_count, "MoE callback missing CPU backend");
        return;
    }
    if ((!ud->experts_registered || !ud->experts || ud->n_experts <= 0) && ud->layer) {
        const densecore::CpuBackend::ExpertWeights* registered_experts = nullptr;
        int registered_count = 0;
        if (ud->backend->GetRegisteredExpertsView(ud->layer, &registered_experts, &registered_count) &&
            registered_experts && registered_count > 0) {
            ud->experts = registered_experts;
            ud->n_experts = registered_count;
            ud->experts_registered = true;
        }
    }
    if (!ud->experts_registered || !ud->experts || ud->n_experts <= 0) {
        FailClosedMoECallback(dst, &g_moe_callback_missing_experts_count, "MoE callback missing registered experts");
        return;
    }

    // Routing (src1 = gate_logits)
    const bool debug_stage_timing = IsMoEStageTimingEnabled();
    const bool collect_profile = ud->profile && profile_begin != std::chrono::steady_clock::time_point{};
    const auto route_begin =
        (debug_stage_timing || collect_profile) ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
    thread_local densecore::moe::MoERouteResult routing;
    const densecore::models::DecoderLayerSpec* layer_spec =
        densecore::models::ResolveDecoderLayerSpecForLayer(ud->model, ud->layer);
    densecore::models::DecoderMoERouter router =
        layer_spec ? layer_spec->ffn.router : densecore::models::DecoderMoERouter::SoftmaxTopK;
    bool use_grouped_sigmoid_routing = router == densecore::models::DecoderMoERouter::GroupedSigmoidTopK;
    if (ForceMoESoftmaxRouting()) {
        use_grouped_sigmoid_routing = false;
        router = densecore::models::DecoderMoERouter::SoftmaxTopK;
    } else if (ForceMoEGroupedSigmoidRouting()) {
        use_grouped_sigmoid_routing = true;
        router = densecore::models::DecoderMoERouter::GroupedSigmoidTopK;
    }
    const bool routed =
        use_grouped_sigmoid_routing
            ? RouteMoEGroupedSigmoid(src1, ud, &routing)
            : (router == densecore::models::DecoderMoERouter::Gemma4SoftmaxTopK ? RouteMoEGemma4TopK(src1, ud, &routing)
                                                                                : RouteMoESoftmaxTopK(src1, ud, &routing));
    if (!routed) {
        FailClosedMoECallback(dst, &g_moe_callback_routing_failure_count, "MoE routing failed");
        return;
    }
    if (ud->test_force_empty_routing) {
        routing.expert_ids.clear();
        routing.weights.clear();
        routing.token_indices.clear();
    }
    if (routing.expert_ids.empty()) {
        FailClosedMoECallback(dst, &g_moe_callback_empty_routing_count, "MoE routing produced no experts");
        return;
    }
    if (ith == 0) {
        DumpMoERouteTrace(src1, ud, routing);
    }
    const auto route_end =
        (debug_stage_timing || collect_profile) ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
    if (ith == 0) {
        UpdateSchedulerExperts(ud, routing);
    }
    const auto scheduler_end =
        debug_stage_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (ud->profile) {
        SetQwen36ProfileMax(ud->profile->moe_task_count, 1);
        int unique_experts = 0;
        for (size_t i = 0; i < routing.expert_ids.size(); ++i) {
            const int expert_id = routing.expert_ids[i];
            bool seen = false;
            for (size_t j = 0; j < i; ++j) {
                if (routing.expert_ids[j] == expert_id) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                ++unique_experts;
            }
        }
        SetQwen36ProfileMax(ud->profile->selected_expert_count, unique_experts);
    }

    // Forward (src0 = input, dst = output)
    densecore::Tensor t_input = GgmlToRowMajorTensor(src0);
    densecore::Tensor t_output = GgmlToRowMajorTensor(dst);
    densecore::CpuBackend::MoEForwardProfile moe_profile;
    densecore::CpuBackend::MoEForwardProfile* moe_profile_ptr = collect_profile ? &moe_profile : nullptr;
    if (collect_profile && route_end >= route_begin) {
        moe_profile.route_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(route_end - route_begin).count());
    }
    const BatchSpec* active_batch = GetCurrentBatch();
    ud->backend->ForwardMoE(ud->model, ud->layer, ud->layer_idx, active_batch, t_input, routing, ud->experts,
                            ud->n_experts, &t_output, moe_profile_ptr);
    if (collect_profile) {
        AddQwen36ProfileNs(
            ud->profile->moe_forward_ns,
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - profile_begin)
                    .count()));
        AddQwen36ProfileNs(ud->profile->moe_route_ns, moe_profile.route_ns);
        AddQwen36ProfileNs(ud->profile->moe_reorder_ns, moe_profile.reorder_ns);
        AddQwen36ProfileNs(ud->profile->moe_expert_ns, moe_profile.expert_ns);
        AddQwen36ProfileNs(ud->profile->moe_reduce_ns, moe_profile.reduce_ns);
        AddQwen36ProfileNs(ud->profile->moe_w1w3_ns, moe_profile.w1w3_ns);
        AddQwen36ProfileNs(ud->profile->moe_w2_ns, moe_profile.w2_ns);
        AddQwen36ProfileNs(ud->profile->moe_rowblock_ns, moe_profile.rowblock_ns);
        AddQwen36ProfileNs(ud->profile->moe_rowblock_w1w3_ns, moe_profile.rowblock_w1w3_ns);
        AddQwen36ProfileNs(ud->profile->moe_rowblock_w2_ns, moe_profile.rowblock_w2_ns);
        ud->profile->moe_decode_scratch_reused.fetch_add(moe_profile.decode_scratch_reused,
                                                         std::memory_order_relaxed);
        ud->profile->moe_decode_allocations_avoided.fetch_add(moe_profile.decode_allocations_avoided,
                                                              std::memory_order_relaxed);
        SetQwen36ProfileMax(ud->profile->moe_rowblock_used, moe_profile.rowblock_used);
        SetQwen36ProfileMax(ud->profile->moe_rowblock_tasks, moe_profile.rowblock_tasks);
    }
    if (debug_stage_timing) {
        const auto backend_end = std::chrono::steady_clock::now();
        const auto route_us =
            std::chrono::duration_cast<std::chrono::microseconds>(route_end - route_begin).count();
        const auto scheduler_us =
            std::chrono::duration_cast<std::chrono::microseconds>(scheduler_end - route_end).count();
        const auto backend_us =
            std::chrono::duration_cast<std::chrono::microseconds>(backend_end - scheduler_end).count();
        std::fprintf(stderr,
                     "[MOE_STAGE] layer=%d batch=%d top_k=%d assignments=%zu route_us=%lld scheduler_us=%lld "
                     "backend_us=%lld\n",
                     ud->layer_idx, routing.batch_size, routing.top_k, routing.expert_ids.size(),
                     static_cast<long long>(route_us), static_cast<long long>(scheduler_us),
                     static_cast<long long>(backend_us));
    }
    DumpMoEOutputTrace(dst, ud);
    if (IsMoEDebugLoggingEnabled()) {
        static std::atomic<int> moe_out_count{0};
        const int call_id = moe_out_count.fetch_add(1);
        if (call_id < 8) {
            DebugLogTensorFiniteStats("MOE_OUT", dst);
        }
    }
}
