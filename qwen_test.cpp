#include "qwen.h"
#include <filesystem>
#include <gtest/gtest.h>

namespace qwen {

namespace fs = std::filesystem;

static inline auto get_num_threads() -> int {
  const char *qwen_num_threads_env = getenv("QWEN_NUM_THREADS");
  int num_threads = qwen_num_threads_env ? std::stoi(qwen_num_threads_env) : get_default_num_threads();
  return num_threads;
}

static inline auto expect_all_close(ggml_tensor *a, ggml_tensor *b, float atol = 1e-5f, float rtol = 0.f) -> void {
  ASSERT_EQ(a->type, b->type);
  ASSERT_EQ(a->type, GGML_TYPE_F32);
  ASSERT_EQ(ggml_nelements(a), ggml_nelements(b));
  int64_t numel = ggml_nelements(a);
  for (int64_t i = 0; i < numel; i++) {
    float ai = ((float *)a->data)[i];
    float bi = ((float *)b->data)[i];
    EXPECT_LT(std::abs(ai - bi), atol + rtol * std::abs(bi)) << "diff " << ai << " vs " << bi;
  }
}

static inline auto read_tensor_data(char *ptr, ggml_tensor *tensor) -> char * {
  memcpy(tensor->data, ptr, ggml_nbytes(tensor));
  return ptr + ggml_nbytes(tensor);
}

// return elapsed time in milliseconds
static inline auto timeit(std::function<void()> fn, int warmup, int active) -> float {
  for (int i = 0; i < warmup; i++) {
    fn();
  }

  int64_t start_us = ggml_time_us();
  for (int i = 0; i < active; i++) {
    fn();
  }
  int64_t end_us = ggml_time_us();

  float elapsed_ms = (end_us - start_us) / 1000.f;
  return elapsed_ms / active;
}

class QwenTest : public ::testing::Test {
  protected:
    ModelContext ctx;

    auto SetUp() -> void override {
      ctx.dtype = GGML_TYPE_F32;
      ctx.ctx_w = make_unique_ggml_context(1024 * MB, nullptr, false);
      ctx.ctx_kv = make_unique_ggml_context(512 * MB, nullptr, false);
      ctx.ctx_b = make_unique_ggml_context(512 * MB, nullptr, false);
      ctx.scratch_buffer.resize(1 * MB);
      ctx.scratch = {0, ctx.scratch_buffer.size(), ctx.scratch_buffer.data()};
      ctx.init_device_context();

      reset_cgraph();
    }

    auto reset_cgraph() -> void { ctx.gf = ggml_new_graph(ctx.ctx_b.get()); }

    auto cpu_graph_compute(int n_threads) -> void { ggml_graph_compute_helper(ctx.work_buffer, ctx.gf, n_threads); }

    auto device_graph_compute(int n_threads) -> void {
      cpu_graph_compute(n_threads);
    }

    template <bool FALLBACK_CPU>
    auto _perf_graph_compute_impl() -> float {
      int num_threads = get_num_threads();
      auto fn = [this, num_threads] {
        if constexpr (FALLBACK_CPU) {
          cpu_graph_compute(num_threads);
        } else {
          device_graph_compute(num_threads);
        }
      };
      return timeit(fn, 1, 3);
    }

    auto perf_cpu_graph_compute() -> float { return _perf_graph_compute_impl<true>(); }
    auto perf_device_graph_compute() -> float { return _perf_graph_compute_impl<false>(); }
};

TEST_F(QwenTest, Embedding) {
  fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/qwe2_0_5b_wte.data";
  MappedFile mapped_file(test_path.string());
  char *ptr = mapped_file.data;

  ggml_tensor *embed_tokens = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 256, 48);
  ptr = read_tensor_data(ptr, embed_tokens);
  ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_I32, 3, 1);
  ptr = read_tensor_data(ptr, x);
  ggml_tensor *y = ggml_new_tensor_3d(ctx.ctx_b.get(), GGML_TYPE_F32, 256, 3, 1);
  ptr = read_tensor_data(ptr, y);
  ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

  // tensor_to_device(x);
  // tensor_to_device(y);

  Embedding m(&ctx, 48, 256);
  m.weight->data = embed_tokens->data;
  // tensor_to_device(m.weight);

  ggml_tensor *out = m.forward(&ctx, x);
  EXPECT_EQ(out->backend, x->backend);
  out->backend = GGML_BACKEND_CPU;

  ggml_build_forward_expand(ctx.gf, out);
  device_graph_compute(get_num_threads());

  expect_all_close(y, out);

  // tensor_to_cpu(m.weight);
  // tensor_to_cpu(y);
  // tensor_to_cpu(x);
}


TEST_F(QwenTest, QwenMLP) {
  fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/qwen2_0_5b_mlp.data";
  MappedFile mapped_file(test_path.string());
  char *ptr = mapped_file.data;

  ggml_tensor *up_proj = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 96);
  ptr = read_tensor_data(ptr, up_proj);
  ggml_tensor *gate_proj = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 96);
  ptr = read_tensor_data(ptr, gate_proj);
  ggml_tensor *down_proj = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 96, 32);
  ptr = read_tensor_data(ptr, down_proj);
  ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 3);
  ptr = read_tensor_data(ptr, x);
  ggml_tensor *ref = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 3);
  ptr = read_tensor_data(ptr, ref);
  ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

  // tensor_to_device(x);
  // tensor_to_device(ref);

  QwenMLP model(&ctx, 32, 96);
  model.up_proj.weight->data = up_proj->data;
  model.gate_proj.weight->data = gate_proj->data;
  model.down_proj.weight->data = down_proj->data;

  // tensor_to_device(model.up_proj.weight);
  // tensor_to_device(model.gate_proj.weight);
  // tensor_to_device(model.down_proj.weight);

  ggml_tensor *out = model.forward(&ctx, x);
  EXPECT_EQ(out->backend, x->backend);
  out->backend = GGML_BACKEND_CPU;

  ggml_build_forward_expand(ctx.gf, out);
  device_graph_compute(get_num_threads());

  expect_all_close(ref, out);

  // tensor_to_cpu(model.up_proj.weight);
  // tensor_to_cpu(model.gate_proj.weight);
  // tensor_to_cpu(model.down_proj.weight);
  // tensor_to_cpu(x);
  // tensor_to_cpu(ref);
}

} // namespace qwen
