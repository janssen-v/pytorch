import torch
from torch._dynamo.utils import counters
import copy
from torch._inductor.fx_passes.post_grad import pass_patterns
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    MultiOutputPattern,
    register_graph_pattern,
)

from torch._inductor.ir import (
    ExternKernel,
    FallbackKernel,
    MultiOutput,
    MultiOutputLayout,
    ExternKernelAlloc,
)
from torch._inductor.lowering import register_lowering, TensorBox
from torch._inductor.utils import sympy_product
from torch._inductor.virtualized import V

import operator

def mindspeed_c2_is_available():
    try:
        # Currently only works when mindspeed is in python path
        from mindspeed.ops.gmm import npu_gmm
        return True
        # Mindspeed all_to_all_v is in a separate "mind" namespace,
        # unsure if it is defined for all mindspeed or just our fork
    except ImportError:
        return False


if mindspeed_c2_is_available():
    aten = torch.ops.aten
    prims = torch.ops.prims
    c10d = torch.ops._c10d_functional
    mindspeed = torch.ops.mind
    mindc2 = torch.ops.c2

    # Register Dummy Torch Ops
    import torch.library
    mixc2_lib = torch.library.Library("mixc2", "DEF")
    mixc2_lib.define(
        "fused_a2agmm(Tensor tokens_local, Tensor tokens_expert, Tensor input_tokens, Tensor perm, Tensor weight, int bucket, str group, int divisor, int zero) -> (Tensor, Tensor)"
    )

    # Reference Kernel Fallback
    def mixc2_fused_a2agmm(
        tokens_local,
        tokens_expert,
        input_tokens,
        perm,
        weight,
        bucket,
        group,
        divisor,
        zero,
    ):
        """Reference Python impl for the fused A2A+GMM pattern.

        This mirrors the matched FX subgraph so functional correctness
        is preserved even when we route scheduling through an extern call.
        """

        # shape helpers
        sum_local = torch.ops.aten.sum.dim_IntList(tokens_local, [-1])
        sum_expert = torch.ops.aten.sum.dim_IntList(tokens_expert, [1])
        sum_axis0 = torch.ops.aten.sum.dim_IntList(tokens_local, [0])
        cumsum = torch.ops.aten.cumsum.default(sum_axis0, 0)

        all_to_all_v = mindspeed.all_to_all_v.default(
            input_tokens, bucket, sum_local, sum_expert, group
        )
        div_floor = aten.div.Tensor_mode(perm, divisor, rounding_mode="floor")
        index = aten.index.Tensor(all_to_all_v, [div_floor])
        gmm = mindc2.npu_gmm.default([index], [weight], [], cumsum, zero, zero)
        return gmm[0], index

    # Pattern 1 (A2AGMM)
    p_tokens_local = Arg()
    p_tokens_expert = Arg()
    p_input_tokens = Arg()
    p_perm = Arg()
    p_weight = Arg()
    p_bucket = Arg()
    p_group = Arg()
    p_divisor = Arg()
    p_zero = Arg()

    sum_local = CallFunction(torch.ops.aten.sum.dim_IntList, p_tokens_local, [-1])
    sum_expert = CallFunction(torch.ops.aten.sum.dim_IntList, p_tokens_expert, [1])
    sum_axis0 = CallFunction(torch.ops.aten.sum.dim_IntList, p_tokens_local, [0])
    cumsum = CallFunction(torch.ops.aten.cumsum.default, sum_axis0, 0)

    all_to_all_v = CallFunction(
        mindspeed.all_to_all_v.default,
        p_input_tokens,
        p_bucket,
        sum_local,
        sum_expert,
        p_group,
    )

    div_floor = CallFunction(
        aten.div.Tensor_mode,
        p_perm,
        p_divisor,
        rounding_mode="floor",
    )

    index_tensor = CallFunction(
        aten.index.Tensor,
        all_to_all_v,
        [div_floor],
    )

    npu_gmm = CallFunction(
        torch.ops.c2.npu_gmm.default,
        [index_tensor],
        [p_weight],
        [],
        cumsum,
        p_zero,
        p_zero,
    )

    getitem0 = CallFunction(operator.getitem, npu_gmm, 0)
    moe_pattern_1 = MultiOutputPattern([getitem0, index_tensor])

    @register_graph_pattern(moe_pattern_1, pass_dict=pass_patterns[2])
    def register_mixc2_fusion_a2agmm(
        match,
        tokens_local,
        tokens_expert,
        input_tokens,
        bucket,
        group,
        perm,
        divisor,
        weight,
        zero,
    ):
        """Rewrite FX graph to call fused A2A+GMM MixC2 Kernel"""
        counters["inductor"]["bishengir_mixc2_fusion_matcher_count"] += 1
        counters["inductor"]["bishengir_mixc2_fusion_matcher_nodes"] += len(match.nodes)

        graph = match.graph
        output_nodes = match.output_nodes()
        if len(output_nodes) != 2 or output_nodes[0] is None or output_nodes[1] is None:
            return  # Safety: unexpected pattern shape; leave graph untouched.

        gmm_out_node, index_node = output_nodes

        # Insert fused call before the first matched output to dominate both.
        anchor = gmm_out_node
        with graph.inserting_before(anchor):
            fused = graph.call_function(
                torch.ops.mixc2.fused_a2agmm,
                (
                    tokens_local,
                    tokens_expert,
                    input_tokens,
                    perm,
                    weight,
                    bucket,
                    group,
                    divisor,
                    zero,
                ),
            )
            fused.meta.update(gmm_out_node.meta)

            fused_gmm = graph.call_function(operator.getitem, (fused, 0))
            fused_idx = graph.call_function(operator.getitem, (fused, 1))

        fused_gmm.meta.update(gmm_out_node.meta)
        fused_idx.meta.update(index_node.meta)

        gmm_out_node.replace_all_uses_with(fused_gmm)
        index_node.replace_all_uses_with(fused_idx)

        match.erase_nodes()

    # @register_lowering(torch.ops.mixc2.fused_a2agmm)
    # def mixc2_fused_a2agmm_lowering(tokens_local, tokens_expert, input_tokens, perm, weight, bucket, group, divisor, zero):

    #     node = V.graph.current_node
    #     vals = getattr(node, "meta", {}).get("val")

    #     if not isinstance(vals, (tuple, list)) or len(vals) != 2:
    #         raise RuntimeError("MixC2 A2AGMM fusion lowering failed to infer output types")

    #     a2agmm_layout = FallbackKernel.tensor_to_layout(vals[0])
    #     index_layout = FallbackKernel.tensor_to_layout(vals[1])

    #     fused = ExternKernelAlloc(
    #         layout=MultiOutputLayout(device=a2agmm_layout.get_device()),
    #         inputs=[tokens_local, tokens_expert, input_tokens, perm, weight],
    #         constant_args=(bucket, group, divisor, zero),
    #         python_kernel_name="torch_npu._inductor.fx_passes.mixc2.mixc2_fused_a2agmm",
    #     )

    #     out0 = MultiOutput(a2agmm_layout, fused, [(tuple, 0)])
    #     out1 = MultiOutput(index_layout, fused, [(tuple, 1)])
    #     fused.outputs = [out0, out1]

    #     return TensorBox.create(out0), TensorBox.create(out1)

    # Pattern 2 (GMMA2A)
    # TBA
