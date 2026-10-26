// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Regression guard for the output-allocator KV-cache invariant on the native
// onnx.Attention path (the com.microsoft.GroupQueryAttention counterpart is
// test_gqa_output_allocator_present_dim.mlir).
//
// Under OGA past_present_share_buffer the host binds past_key_values.* and
// present.* to the same max_length capacity buffer and the attention mask
// spans only the valid keys. GetOutput returns that pre-bound OrtValue only if
// the in-graph present shape equals it, so present must be sized
// max(dim(past_key, 2), total) and not past + current, which overshoots the
// buffer by the new tokens and fails the allocator callback. The total KV
// length (seqlens_k + 1) is min(past + current, mask key extent): the mask
// extent for a shared buffer, past + current for a concatenated cache, where
// the max then also yields past + current.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-to-hip-pipeline 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME:    %[[K:[a-zA-Z0-9_]+]]: memref<1x?x2048xf16> {onnx.name = "k"}
// CHECK-SAME:    %[[MASK:[a-zA-Z0-9_]+]]: memref<1x1x?x?xf16> {onnx.name = "attn_mask"}
// CHECK-SAME:    %[[PK:[a-zA-Z0-9_]+]]: memref<1x8x?x256xf16> {onnx.name = "past_key_values.0.key"}
// CHECK:         %[[CUR:.*]] = memref.dim %[[K]], %{{.*}} : memref<1x?x2048xf16>
// CHECK:         %[[PAST:.*]] = memref.dim %[[PK]], %{{.*}} : memref<1x8x?x256xf16>
// CHECK:         %[[SUM:.*]] = arith.addi %[[PAST]], %[[CUR]] : index
// CHECK:         %[[KEYS:.*]] = memref.dim %[[MASK]], %{{.*}} : memref<1x1x?x?xf16>
// CHECK:         %[[TOT:.*]] = arith.minui %[[SUM]], %[[KEYS]] : index
// CHECK:         %[[CAP:.*]] = arith.maxui %[[PAST]], %[[TOT]] : index
// CHECK:         arith.subi %[[TOT]], %{{.*}} : index
// CHECK:         hip.alloc_output(%{{.*}}, %[[CAP]]) {out_idx = 0 : i64} : memref<1x8x?x256xf16>
// CHECK:         hip.alloc_output(%{{.*}}, %[[CAP]]) {out_idx = 1 : i64} : memref<1x8x?x256xf16>

module {
  func.func @main_graph(
      %q: tensor<1x?x4096xf16> {onnx.name = "q"},
      %k: tensor<1x?x2048xf16> {onnx.name = "k"},
      %v: tensor<1x?x2048xf16> {onnx.name = "v"},
      %mask: tensor<1x1x?x?xf16> {onnx.name = "attn_mask"},
      %past_key: tensor<1x8x?x256xf16> {onnx.name = "past_key_values.0.key"},
      %past_value: tensor<1x8x?x256xf16> {onnx.name = "past_key_values.0.value"})
      -> (tensor<1x8x?x256xf16> {onnx.name = "present.0.key"},
          tensor<1x8x?x256xf16> {onnx.name = "present.0.value"}) {
    %r:3 = "onnx.Attention"(%q, %k, %v, %mask, %past_key, %past_value)
        {q_num_heads = 16 : si64, kv_num_heads = 8 : si64, is_causal = 0 : si64,
         scale = 1.000000e+00 : f32, softcap = 0.000000e+00 : f32}
        : (tensor<1x?x4096xf16>, tensor<1x?x2048xf16>, tensor<1x?x2048xf16>,
           tensor<1x1x?x?xf16>, tensor<1x8x?x256xf16>, tensor<1x8x?x256xf16>)
        -> (tensor<1x?x4096xf16>, tensor<1x8x?x256xf16>, tensor<1x8x?x256xf16>)
    "onnx.Return"(%r#1, %r#2) : (tensor<1x8x?x256xf16>, tensor<1x8x?x256xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
