#include "qwen.h"
#include "base64.h"
#include "unordered_dense.h"
#include <fcntl.h>
#include <fstream>
#include <numeric>
#include <random>
#include <thread>
#include <sys/stat.h>
#include <iostream>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <stdio.h>
#include <windows.h>
#endif

namespace qwen {

ggml_tensor *tensor_assign_buffers(ggml_tensor *tensor) {
#ifdef GGML_USE_CUBLAS
  ggml_cuda_assign_buffers(tensor);
#endif
  return tensor;
}

auto tensor_to_device(ggml_tensor *tensor) -> ggml_tensor * {
#ifdef GGML_USE_CUBLAS
  if (tensor->backend == GGML_BACKEND_CPU) {
    tensor->backend = GGML_BACKEND_GPU;
    ggml_cuda_transform_tensor(tensor->data, tensor);
  }
#endif
  return tensor;
}

auto tensor_to_cpu(ggml_tensor *tensor) -> ggml_tensor * {
#ifdef GGML_USE_CUBLAS
  if (tensor->backend != GGML_BACKEND_CPU) {
    ggml_cuda_free_data(tensor);
    tensor->backend = GGML_BACKEND_CPU;
  }
#endif
  return tensor;
}

const std::string ChatMessage::ROLE_USER = "user";
const std::string ChatMessage::ROLE_ASSISTANT = "assistant";
const std::string ChatMessage::ROLE_SYSTEM = "system";

void QwenTokenizer::check_chat_messages(const std::vector<ChatMessage> &messages) {
   // default we have system param
    QWEN_CHECK(messages.size() % 2 == 0) << "invalid chat messages size " << messages.size();
    for (size_t i = 1; i < messages.size(); i++) {
        const std::string &target_role = (i % 2 == 1) ? ChatMessage::ROLE_USER : ChatMessage::ROLE_ASSISTANT;
        QWEN_CHECK(messages[i].role == target_role)
            << "expect messages[" << i << "].role to be " << target_role << ", but got " << messages[i].role;
    }
}

// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/llama.cpp
auto ggml_graph_compute_helper(std::vector<uninitialized_char> &buf, ggml_cgraph *graph, int n_threads) -> void {
  struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

  if (plan.work_size > 0) {
    buf.resize(plan.work_size);
    plan.work_data = (uint8_t *)buf.data();
  }

  ggml_graph_compute(graph, &plan);
}

auto ModelContext::init_device_context() -> void {
#ifdef GGML_USE_METAL
  ctx_metal = make_unique_ggml_metal_context(1);
  const size_t max_size = ggml_get_max_tensor_size(ctx_w.get());
  void *weight_data = weight_buffer.empty() ? ggml_get_mem_buffer(ctx_w.get()) : (void *)weight_buffer.data();
  size_t weight_size = weight_buffer.empty() ? ggml_get_mem_size(ctx_w.get()) : weight_buffer.size();
  QWEN_CHECK(ggml_metal_add_buffer(ctx_metal.get(), "weights", weight_data, weight_size, max_size));
  QWEN_CHECK(ggml_metal_add_buffer(ctx_metal.get(), "kv", ggml_get_mem_buffer(ctx_kv.get()),
             ggml_get_mem_size(ctx_kv.get()), 0));
  void *compute_data = ctx_b ? ggml_get_mem_buffer(ctx_b.get()) : compute_buffer.data();
  size_t compute_size = ctx_b ? ggml_get_mem_size(ctx_b.get()) : compute_buffer.size();
  QWEN_CHECK(ggml_metal_add_buffer(ctx_metal.get(), "compute", compute_data, compute_size, 0));
  QWEN_CHECK(ggml_metal_add_buffer(ctx_metal.get(), "scratch", scratch.data, scratch.size, 0));
#endif
}

// ===== streamer =====

auto StreamerGroup::put(const std::vector<int_least32_t> &output_ids) -> void {
  for (auto &streamer : streamers_) {
    streamer->put(output_ids);
  }
}

auto StreamerGroup::end() -> void {
  for (auto &streamer : streamers_) {
    streamer->end();
  }
}

auto TextStreamer::put(const std::vector<int> &output_ids) -> void {
  if (is_prompt_) {
    is_prompt_ = false;
    return;
  }

  static const std::vector<char> puncts{',', '!', ':', ';', '?'};

  token_cache_.insert(token_cache_.end(), output_ids.begin(), output_ids.end());
  std::string text = tokenizer_->decode(token_cache_);
  if (text.empty()) {
    return;
  }

  std::string printable_text;
  if (text.back() == '\n') {
    // flush the cache after newline
    printable_text = text.substr(print_len_);
    token_cache_.clear();
    print_len_ = 0;
  } else if (std::find(puncts.begin(), puncts.end(), text.back()) != puncts.end()) {
    // last symbol is a punctuation, hold on
  } else if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
    // ends with an incomplete token, hold on
  } else {
    printable_text = text.substr(print_len_);
    print_len_ = text.size();
  }

  os_ << printable_text << std::flush;
}

auto TextStreamer::end() -> void {
  std::string text = tokenizer_->decode(token_cache_);
  os_ << text.substr(print_len_) << std::endl;
  is_prompt_ = true;
  token_cache_.clear();
  print_len_ = 0;
}

auto PerfStreamer::put(const std::vector<int> &output_ids) -> void {
  QWEN_CHECK(!output_ids.empty());
  if (num_prompt_tokens_ == 0) {
    // before prompt eval
    start_us_ = ggml_time_us();
    num_prompt_tokens_ = output_ids.size();
  } else {
    if (num_output_tokens_ == 0) {
      // first new token
      prompt_us_ = ggml_time_us();
    }
    num_output_tokens_ += output_ids.size();
  }
}

auto PerfStreamer::reset() -> void {
  start_us_ = prompt_us_ = end_us_ = 0;
  num_prompt_tokens_ = num_output_tokens_ = 0;
}

auto PerfStreamer::to_string() -> std::string const {
  std::ostringstream oss;
  oss << "prompt time: " << prompt_total_time_us() / 1000.f << " ms / " << num_prompt_tokens() << " tokens ("
    << prompt_token_time_us() / 1000.f << " ms/token)\n"
    << "output time: " << output_total_time_us() / 1000.f << " ms / " << num_output_tokens() << " tokens ("
    << output_token_time_us() / 1000.f << " ms/token)\n"
    << "total time: " << (prompt_total_time_us() + output_total_time_us()) / 1000.f << " ms";
  return oss.str();
}


#if !defined(_WIN32)
MappedFile::MappedFile(const std::string &path) {
  int fd = open(path.c_str(), O_RDONLY);
  QWEN_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

  struct stat sb;
  QWEN_CHECK(fstat(fd, &sb) == 0) << strerror(errno);
  size = sb.st_size;

  data = (char *)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
  QWEN_CHECK(data != MAP_FAILED) << strerror(errno);

  QWEN_CHECK(close(fd) == 0) << strerror(errno);
}

MappedFile::~MappedFile() { QWEN_CHECK(munmap(data, size) == 0) << strerror(errno); }
#else
MappedFile::MappedFile(const std::string& path) {

    int fd = open(path.c_str(), O_RDONLY);
    QWEN_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

    struct _stat64 sb;
    QWEN_CHECK(_fstat64(fd, &sb) == 0) << strerror(errno);
    size = sb.st_size;

    HANDLE hFile = (HANDLE)_get_osfhandle(fd);

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    QWEN_CHECK(hMapping != NULL) << strerror(errno);

    data = (char*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);

    QWEN_CHECK(data != NULL) << strerror(errno);

    QWEN_CHECK(close(fd) == 0) << strerror(errno);
}

MappedFile::~MappedFile() { QWEN_CHECK(UnmapViewOfFile(data)) << strerror(errno); }

#endif

auto ModelLoader::seek(int64_t offset, int whence) -> void {
  if (whence == SEEK_SET) {
    ptr = data + offset;
  } else if (whence == SEEK_CUR) {
    ptr += offset;
  } else if (whence == SEEK_END) {
    ptr = data + size + offset;
  } else {
    QWEN_THROW << "invalid seek mode " << whence;
  }
}

auto ModelLoader::read_string(size_t length) -> std::string {
  std::string s(ptr, ptr + length);
  ptr += length;
  return s;
}

auto ModelLoader::read_tensor(const std::string &name, ggml_tensor *tensor) -> void {
  // read and check tensor name
  {
    int name_size = read_basic<int>();
    QWEN_CHECK(name_size == (int)name.size())
      << "tensor " << name << " name size mismatch: expect " << name.size() << " but got " << name_size;
    std::string weight_name = read_string(name_size);
    QWEN_CHECK(weight_name == name) << "tensor name mismatch: expect " << name << " but got " << weight_name;
  }

  // read and check tensor shape
  {
    int ndim = read_basic<int>();
    int n_dims = ggml_n_dims((tensor));
      // a quick fix
      if ((n_dims == 1) && (ndim == 2) && (tensor->ne[1] == 1))
          n_dims = 2;
    QWEN_CHECK(ndim == n_dims)
      << "tensor " << name << " ndim mismatch: expect " << n_dims << " but got " << ndim;
    for (int i = ndim - 1; i >= 0; i--) {
      int dim_size = read_basic<int>();
      QWEN_CHECK(dim_size == tensor->ne[i]) << "tensor " << name << " shape mismatch at dim " << i
                                            << ": expect " << tensor->ne[i] << " but got " << dim_size;
    }
  }

  // read and check tensor dtype
  {
    ggml_type dtype = (ggml_type)read_basic<int>();
    QWEN_CHECK(dtype == tensor->type)
      << "tensor " << name << " dtype mismatch: expect " << tensor->type << " but got " << dtype;
  }

  // map tensor data
  {
    constexpr int64_t MEM_ALIGNED = 16;
    const int64_t data_offset = (tell() + (MEM_ALIGNED - 1)) & ~(MEM_ALIGNED - 1);
    tensor->data = const_cast<char *const>(data) + data_offset;
    seek(data_offset + ggml_nbytes(tensor), SEEK_SET);
  }
}

// ===== modules =====

auto Embedding::forward(ModelContext *ctx, ggml_tensor *input) const -> ggml_tensor * {
  ggml_tensor *output = (ggml_n_dims(input) == 1) && (ggml_type::GGML_TYPE_I32 == input->type)
                          ? ggml_get_rows(ctx->ctx_b.get(), weight, input)
                          : ggml_mul_mat(ctx->ctx_b.get(), weight, input);
  return output;
}

auto Linear::forward(ModelContext *ctx, ggml_tensor *input) const -> ggml_tensor * {
  // input: [seqlen, in_features]
  ggml_context *gctx = ctx->ctx_b.get();
  ggml_tensor *output = tensor_assign_buffers(ggml_mul_mat(gctx, weight, input)); // [seqlen, out_features]
  if (bias) {
    output = tensor_assign_buffers(ggml_add_inplace(gctx, output, bias));
  }
  return output;
}

auto RMSNorm::forward(ModelContext *ctx, ggml_tensor *input, float eps) const -> ggml_tensor * {
  ggml_context *gctx = ctx->ctx_b.get();
  auto ggml_rms_norm_fn = inplace ? ggml_rms_norm_inplace : ggml_rms_norm;
  ggml_tensor *output = tensor_assign_buffers(ggml_rms_norm_fn(gctx, input, eps));
  output = tensor_assign_buffers(ggml_mul_inplace(gctx, output, weight));
  return output;
}

// ===== Qwen =====

static std::pair<std::string, int> _parse(const std::string &line) {
  auto pos = line.find(" ");
  if (pos == std::string::npos) {
    throw std::runtime_error("invalid encoder line: " + line);
  }

  auto token = base64::decode({line.data(), pos});
  int rank = 0;
  try {
    rank = std::stoul(line.substr(pos + 1));
  } catch (const std::exception &) {
    throw std::runtime_error("invalid encoder rank: " + line);
  }

  return {std::move(token), rank};
}

QwenTokenizer::QwenTokenizer(const std::string & tiktoken_path, const QwenConfig &config) {
  std::ifstream file(tiktoken_path);
  if (!file) {
    throw std::runtime_error("failed to open encoder file: " + tiktoken_path);
  }

  ankerl::unordered_dense::map<std::string, int> encoder;
  std::string line;
  while (std::getline(file, line)) {
    auto [token, rank] = _parse(line);

    if (!encoder.emplace(std::move(token), rank).second) {
      throw std::runtime_error("duplicate item: " + line);
    }
  }

  std::vector<std::string> special_tokens_s{"<|endoftext|>", "<|im_start|>", "<|im_end|>"};
  char buffer[14];
  for (size_t i = 0; i < 205; i++) { // 205 for extra control token
    snprintf(buffer, 14, "<|extra_%zu|>", i);
    special_tokens_s.push_back(buffer);
  }
  size_t encoder_size = encoder.size();
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  special_tokens.reserve(special_tokens_s.size());
  for (size_t i = 0; i < special_tokens_s.size(); i++) {
    special_tokens[special_tokens_s[i]] = encoder_size + i;
  }

  tokenizer = tiktoken::tiktoken(std::move(encoder), special_tokens, PAT_STR);
  eos_token_id = config.eos_token_id;
  im_start_id = config.im_start_id;
  im_end_id = config.im_end_id;
}


auto QwenTokenizer::encode(const std::string &text, int max_length) const -> std::vector<int> {
  auto ids = tokenizer.encode(text);
  if ((int)ids.size() > max_length) {
    ids.erase(ids.begin(), ids.end() - max_length);
  }
  return ids;
}

auto QwenTokenizer::decode(const std::vector<int> &ids) const -> std::string {
  std::vector<int> normal_ids(ids);
  normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id) { return is_special_id(id); }),
                   normal_ids.end());
  auto text = tokenizer.decode(normal_ids);
  return text;
}

std::string QwenTokenizer::build_prompt(const std::vector<ChatMessage> &messages) {

  // check_chat_messages(messages); 

  std::ostringstream oss_prompt;
  oss_prompt << "<|im_start|>system\n" << messages.front().content << "<|im_end|>"; // apply system
  
  for (size_t i = 1; i < messages.size(); i += 1) { // get all history messages
    if(messages[i].role == ChatMessage::ROLE_USER){
      oss_prompt << "\n<|im_start|>user\n" << messages[i].content << "<|im_end|>\n<|im_start|>assistant\n";
    }else{
      oss_prompt << messages[i].content << "<|im_end|>";
    }
  }

  // oss_prompt << "\n<|im_start|>user\n" << messages.back().content <<  "<|im_end|>\n<|im_start|>assistant\n"; // old way: last message

  // std::cout << oss_prompt.str()<<std::endl;

  return oss_prompt.str();
}

std::vector<int> QwenTokenizer::encode_messages(const std::vector<ChatMessage> &messages, int max_length) const {
    std::string prompt = build_prompt(messages);
    std::vector<int> input_ids = encode(prompt, max_length);
    return input_ids;
}

auto QwenTokenizer::is_special_id(int id) const -> bool {
  return id == eos_token_id || id == im_start_id || id == im_end_id;
}

QwenAttention::QwenAttention(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
  : num_attention_heads(num_attention_heads), num_kv_heads(num_kv_heads),
    q_proj(ctx, hidden_size, hidden_size), // head_dim * num_attention_heads
    k_proj(ctx, hidden_size, hidden_size/(num_attention_heads/num_kv_heads)), // head_dim * num_kv_heads. Qwen1.5 32B is GQA model
    v_proj(ctx, hidden_size, hidden_size/(num_attention_heads/num_kv_heads)), // head_dim * num_kv_heads
    o_proj(ctx, hidden_size, hidden_size, false),
    k_cache(ggml_new_tensor_3d(ctx->ctx_kv.get(), GGML_TYPE_F16, hidden_size / num_attention_heads, max_length,
                               num_kv_heads)),
    v_cache(ggml_new_tensor_3d(ctx->ctx_kv.get(), GGML_TYPE_F16, max_length, hidden_size / num_attention_heads,
                               num_kv_heads)) {}

auto QwenAttention::forward(ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor * {
  ggml_context *gctx = ctx->ctx_b.get();

  const int hidden_size = hidden_states->ne[0];
  const int qlen = hidden_states->ne[1];
  const int head_size = hidden_size / num_attention_heads;
  const int rope_dim = head_size;
  const int n_past = static_cast<int *>(KQ_pos->data)[0];

  ggml_tensor *q = q_proj.forward(ctx, hidden_states); // [qlen, hidden]
  // [qlen, heads, head_size]
  ggml_tensor *query_layer = ggml_view_3d(gctx, q, head_size, num_attention_heads, qlen, head_size * ggml_element_size(q), q->nb[1], 0);
 
  ggml_tensor *k = k_proj.forward(ctx, hidden_states); // [qlen, kv_hidden]
  // [qlen, kv_heads, head_size]
  ggml_tensor *key_layer = ggml_view_3d(gctx, k, head_size, num_kv_heads, qlen, head_size * ggml_element_size(k), k->nb[1], 0);

  ggml_tensor *v = v_proj.forward(ctx, hidden_states); // [qlen, kv_hidden]
  ggml_tensor *value_layer = ggml_view_3d(gctx, v, head_size, num_kv_heads, qlen, head_size * ggml_element_size(v), v->nb[1], 0);

#ifdef GGML_USE_CUBLAS
  if (!ggml_is_contiguous(query_layer)) {
    query_layer = tensor_assign_buffers(ggml_cont(gctx, query_layer));
  }
#endif
  query_layer = tensor_assign_buffers(ggml_rope_inplace(gctx, query_layer, KQ_pos, rope_dim, 2, n_ctx));
  query_layer = tensor_assign_buffers(ggml_permute(gctx, query_layer, 0, 2, 1, 3)); // [heads, qlen, head_size]

#ifdef GGML_USE_CUBLAS
  if (!ggml_is_contiguous(key_layer)) {
    key_layer = tensor_assign_buffers(ggml_cont(gctx, key_layer));
  }
#endif
  key_layer = tensor_assign_buffers(ggml_rope_inplace(gctx, key_layer, KQ_pos, rope_dim, 2, n_ctx));
  key_layer = tensor_assign_buffers(ggml_permute(gctx, key_layer, 0, 2, 1, 3)); // [kv_heads, qlen, head_size]
  value_layer = tensor_assign_buffers(ggml_permute(gctx, value_layer, 1, 2, 0, 3)); // [kv_heads, head_size, qlen]

  // store key & value to cache
  ggml_tensor *k_cache_view = tensor_assign_buffers(
    ggml_view_3d(gctx, k_cache, head_size, qlen, num_kv_heads, k_cache->nb[1], k_cache->nb[2],
                 n_past * head_size * ggml_element_size(k_cache))); // [kv_heads, qlen, head_size]
  ggml_build_forward_expand(ctx->gf, ggml_cpy(gctx, key_layer, k_cache_view));
  ggml_tensor *v_cache_view = tensor_assign_buffers(
    ggml_view_3d(gctx, v_cache, qlen, head_size, num_kv_heads, v_cache->nb[1], v_cache->nb[2],
                 n_past * ggml_element_size(v_cache))); // [kv_heads, head_size, qlen]
  ggml_build_forward_expand(ctx->gf, ggml_cpy(gctx, value_layer, v_cache_view));  

  // concat key & value with past kv
  key_layer = tensor_assign_buffers(
    ggml_view_3d(gctx, k_cache, head_size, n_past + qlen, num_kv_heads,
                 k_cache->nb[1], k_cache->nb[2], 0)); // [kv_heads, klen, head_size]
  value_layer = tensor_assign_buffers(
    ggml_view_3d(gctx, v_cache, n_past + qlen, head_size, num_kv_heads,
                 v_cache->nb[1], v_cache->nb[2], 0)); // [kv_heads, head_size, klen]

  // attention
  ggml_tensor *attn_scores = 
    tensor_assign_buffers(ggml_mul_mat(gctx, key_layer, query_layer)); // [kv_heads, mqa_scale * qlen, klen]
  attn_scores = tensor_assign_buffers(
    ggml_scale_inplace(gctx, attn_scores, 1.f / std::sqrt(head_size)));
  if (n_past == 0) {
    // build attention mask for context input
    attn_scores = tensor_assign_buffers(ggml_diag_mask_inf_inplace(gctx, attn_scores, n_past));
  }
  ggml_tensor *attn_probs =
    tensor_assign_buffers(ggml_soft_max_inplace(gctx, attn_scores)); // [kv_heads, mqa_scale * qlen, klen]

  ggml_tensor *context_layer = tensor_assign_buffers(
    ggml_mul_mat(gctx, value_layer, attn_probs)); // [kv_heads, mqa_scale * qlen, head_size]
  context_layer = tensor_assign_buffers(
    ggml_cont(gctx, ggml_permute(gctx, context_layer, 0, 2, 1, 3))); // [qlen, heads, head_size]
  context_layer = tensor_assign_buffers(
    ggml_reshape_2d(gctx, context_layer, hidden_size, qlen)); // [qlen, hidden]

  ggml_tensor *attn_output = o_proj.forward(ctx, context_layer);
  return attn_output;
}

auto QwenMLP::forward(ModelContext *ctx, ggml_tensor *hidden_states) const -> ggml_tensor * {
  ggml_context *gctx = ctx->ctx_b.get();

  ggml_tensor *gate_out = gate_proj.forward(ctx, hidden_states);
  gate_out = tensor_assign_buffers(ggml_silu_inplace(gctx, gate_out));
  ggml_tensor *up_out = up_proj.forward(ctx, hidden_states);

  ggml_tensor *output = tensor_assign_buffers(ggml_mul_inplace(gctx, gate_out, up_out));
  output = down_proj.forward(ctx, output);
  return output;
}

auto Qwen2MoeSparseMoeBlock::forward(ModelContext *ctx, ggml_tensor *hidden_states, int num_experts, int num_experts_per_tok) const -> ggml_tensor *{
  const int64_t n_tokens = hidden_states->ne[1];
  const int n_expert = num_experts;

  ggml_context *gctx = ctx->ctx_b.get();

  ggml_tensor *logits = gate.forward(ctx, hidden_states); // [n_tokens, num_experts]
  ggml_tensor *probs = ggml_soft_max(gctx, logits); // [n_tokens, num_experts]

  // select experts
  ggml_tensor * selected_experts = ggml_top_k(gctx, probs, num_experts_per_tok); // [n_tokens, num_experts_per_tok]

  ggml_tensor * weights = ggml_get_rows(gctx, 
      ggml_reshape_3d(gctx, probs, 1, n_expert, n_tokens), selected_experts);

  weights = ggml_reshape_2d(gctx, weights, num_experts_per_tok, n_tokens);  // [n_tokens, num_experts_per_tok]

  if(norm_topk_prob)
  {
      ggml_tensor * weights_sum = ggml_sum_rows(gctx, weights);
      weights = ggml_div(gctx, weights, weights_sum); // [n_tokens, num_experts_per_tok]
  }

  // compute expert outputs
  ggml_tensor *moe_out = nullptr;

  for(int i = 0; i < num_experts_per_tok; i++)
  {
      ggml_tensor * cur_expert;
      ggml_tensor * cur_up = ggml_mul_mat_id(gctx, expert_ups.data(), n_expert, selected_experts, i, hidden_states);
      ggml_tensor * cur_gate = ggml_mul_mat_id(gctx, expert_gates.data(), n_expert, selected_experts, i, hidden_states);

      cur_gate = ggml_silu(gctx, cur_gate);
      cur_expert = ggml_mul(gctx, cur_up, cur_gate); // [n_tokens, n_embd]
      cur_expert = ggml_mul_mat_id(gctx, expert_downs.data(), n_expert, selected_experts, i, cur_expert); // [n_tokens, n_embd]

      cur_expert = ggml_mul(gctx, cur_expert,
                  ggml_view_2d(gctx, weights, 1, n_tokens, weights->nb[1], i * weights->nb[0]));
      if (i == 0){
        moe_out = cur_expert;
      }else{
        moe_out = ggml_add(gctx, moe_out, cur_expert);
      }
  }

  // shared expert
  ggml_tensor * shared_expert_output = shared_expert.forward(ctx, hidden_states);
  ggml_tensor * shared_expert_gate_output = shared_expert_gate.forward(ctx, hidden_states);

  ggml_tensor * logits_shared_exp = ggml_silu(gctx, shared_expert_gate_output); // sigmoid
  shared_expert_gate_output = ggml_div(gctx, logits_shared_exp, shared_expert_gate_output);

  ggml_tensor * final_hidden_states = ggml_mul(gctx, shared_expert_output , shared_expert_gate_output);

  final_hidden_states = ggml_add(gctx, final_hidden_states, moe_out);
  
  return final_hidden_states;
}

auto QwenBlock::forward(ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor * {
  ggml_context *gctx = ctx->ctx_b.get();

  ggml_tensor *residual = hidden_states;
  hidden_states = input_layernorm.forward(ctx, hidden_states, 1e-6f);
  hidden_states = attn.forward(ctx, hidden_states, KQ_pos, n_ctx); // FAILE HERE  a->ne[2] == b->ne[0]
  hidden_states = tensor_assign_buffers(ggml_add_inplace(gctx, hidden_states, residual));

  residual = hidden_states;
  hidden_states = post_attention_layernorm.forward(ctx, hidden_states, 1e-6f);
  hidden_states = mlp.forward(ctx, hidden_states);
  hidden_states = tensor_assign_buffers(ggml_add_inplace(gctx, hidden_states, residual));
  return hidden_states;
}

auto QwenMoeBlock::forward(ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *KQ_pos, int n_ctx , int num_experts, int num_experts_per_tok) const -> ggml_tensor * {
  ggml_context *gctx = ctx->ctx_b.get();

  ggml_tensor *residual = hidden_states;
  hidden_states = input_layernorm.forward(ctx, hidden_states, 1e-6f);
  hidden_states = attn.forward(ctx, hidden_states, KQ_pos, n_ctx); // FAILE HERE  a->ne[2] == b->ne[0]

  hidden_states = tensor_assign_buffers(ggml_add_inplace(gctx, hidden_states, residual));

  residual = hidden_states;
  hidden_states = post_attention_layernorm.forward(ctx, hidden_states, 1e-6f);
  hidden_states = mlp.forward(ctx, hidden_states, num_experts, num_experts_per_tok);
  hidden_states = tensor_assign_buffers(ggml_add_inplace(gctx, hidden_states, residual));
  return hidden_states;
}

QwenModel::QwenModel(ModelContext *ctx, const QwenConfig &config)
  : embed_tokens(ctx, config.vocab_size, config.hidden_size), norm(ctx, config.hidden_size) {
  layers.reserve(config.num_hidden_layers);
  for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++) {
    layers.emplace_back(ctx, config.hidden_size, config.num_attention_heads, config.num_kv_heads,
      config.intermediate_size, config.max_length);
  }
}

auto QwenModel::forward(ModelContext *ctx, ggml_tensor *input_ids, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor * {
  ggml_context *gctx = ctx->ctx_b.get();
  ggml_tensor *hidden_states = embed_tokens.forward(ctx, input_ids);
  for (const auto &layer : layers) {
    ggml_set_scratch(gctx, ctx->scratch);
    hidden_states = layer.forward(ctx, hidden_states, KQ_pos, n_ctx);
  }
  ggml_scratch empty_scratch = {0, 0, nullptr};
  ggml_set_scratch(gctx, empty_scratch);
  hidden_states = norm.forward(ctx, hidden_states, 1e-6f);
  return hidden_states;
}

QwenMoeModel::QwenMoeModel(ModelContext *ctx, const QwenMoeConfig &config)
  : embed_tokens(ctx, config.vocab_size, config.hidden_size), norm(ctx, config.hidden_size) {
  layers.reserve(config.num_hidden_layers);
  for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++) {
    layers.emplace_back(ctx, config.hidden_size, config.num_attention_heads, config.num_kv_heads,
      config.intermediate_size, config.moe_intermediate_size, config.shared_expert_intermediate_size, config.num_experts, config.num_experts_per_tok, config.max_length);
  }
}

auto QwenMoeModel::forward(ModelContext *ctx, ggml_tensor *input_ids, ggml_tensor *KQ_pos, int n_ctx,  int num_experts, int num_experts_per_tok) const -> ggml_tensor * {
  ggml_context *gctx = ctx->ctx_b.get();
  ggml_tensor *hidden_states = embed_tokens.forward(ctx, input_ids);
  for (const auto &layer : layers) {
    ggml_set_scratch(gctx, ctx->scratch);
    hidden_states = layer.forward(ctx, hidden_states, KQ_pos, n_ctx, num_experts, num_experts_per_tok);
  }
  ggml_scratch empty_scratch = {0, 0, nullptr};
  ggml_set_scratch(gctx, empty_scratch);
  hidden_states = norm.forward(ctx, hidden_states, 1e-6f);
  return hidden_states;
}

// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp
auto get_num_physical_cores() -> int {
  unsigned int n_threads = std::thread::hardware_concurrency();
  return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

auto get_default_num_threads() -> int {
#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_METAL)
    return 1;
#else
  return std::min(get_num_physical_cores(), 16);
#endif
}

QwenForCausalLM::QwenForCausalLM(const QwenConfig &config)
  : config(config) {
  ctx_.compute_buffer.resize(MEM_SIZE);
  ctx_.scratch_buffer.resize(SCRATCH_SIZE);
  ctx_.scratch = {0, ctx_.scratch_buffer.size(), ctx_.scratch_buffer.data()};
#ifdef GGML_USE_CUBLAS
  ggml_cuda_set_scratch_size(SCRATCH_SIZE);
#endif
  constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
  const size_t ctx_w_size = (3 + config.num_hidden_layers * 12) * tensor_ovhd;
  const size_t ctx_kv_size = 2 * config.num_hidden_layers *
                             (config.max_length * config.hidden_size / config.num_attention_heads * config.num_kv_heads * ggml_type_size(GGML_TYPE_F16) + tensor_ovhd);
  ctx_.dtype = config.dtype;
  ctx_.ctx_w = make_unique_ggml_context(ctx_w_size, nullptr, true);
  ctx_.ctx_kv = make_unique_ggml_context(ctx_kv_size + 1 * MB, nullptr, false); // 1MB extra for MPS

  transformer = QwenModel(&ctx_, config);
  lm_head = Linear(&ctx_, config.hidden_size, config.vocab_size, false);
  QWEN_CHECK(ggml_used_mem(ctx_.ctx_w.get()) == ggml_get_mem_size(ctx_.ctx_w.get())) << "corrupted model weights";
  QWEN_CHECK(ggml_used_mem(ctx_.ctx_kv.get()) == ctx_kv_size) << "corrupted kv cache";

  // build state_dict
  state_dict_.reserve(3 + config.num_hidden_layers * 12);
  state_dict_.emplace_back("model.embed_tokens.weight", transformer.embed_tokens.weight);
  for (int i = 0; i < config.num_hidden_layers; i++) {
    std::string layer_prefix = "model.layers." + std::to_string(i) + '.';
    state_dict_.emplace_back(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);
    
    state_dict_.emplace_back(layer_prefix + "self_attn.q_proj.weight",
                             transformer.layers[i].attn.q_proj.weight);
    state_dict_.emplace_back(layer_prefix + "self_attn.q_proj.bias",
                             transformer.layers[i].attn.q_proj.bias);

    state_dict_.emplace_back(layer_prefix + "self_attn.k_proj.weight",
                             transformer.layers[i].attn.k_proj.weight);
    state_dict_.emplace_back(layer_prefix + "self_attn.k_proj.bias",
                             transformer.layers[i].attn.k_proj.bias);

    state_dict_.emplace_back(layer_prefix + "self_attn.v_proj.weight",
                             transformer.layers[i].attn.v_proj.weight);
    state_dict_.emplace_back(layer_prefix + "self_attn.v_proj.bias",
                             transformer.layers[i].attn.v_proj.bias);

    state_dict_.emplace_back(layer_prefix + "self_attn.o_proj.weight",
                             transformer.layers[i].attn.o_proj.weight);

    state_dict_.emplace_back(layer_prefix + "post_attention_layernorm.weight",
                             transformer.layers[i].post_attention_layernorm.weight);
    state_dict_.emplace_back(layer_prefix + "mlp.gate_proj.weight",
                             transformer.layers[i].mlp.gate_proj.weight);
    state_dict_.emplace_back(layer_prefix + "mlp.up_proj.weight",
                             transformer.layers[i].mlp.up_proj.weight);
    state_dict_.emplace_back(layer_prefix + "mlp.down_proj.weight",
                             transformer.layers[i].mlp.down_proj.weight);
  }
  state_dict_.emplace_back("model.norm.weight", transformer.norm.weight);
  state_dict_.emplace_back("lm_head.weight", lm_head.weight);

}

QwenMoeForCausalLM::QwenMoeForCausalLM(const QwenMoeConfig &config)
  : QwenForCausalLM(config), config(config) {

  ctx_.compute_buffer.resize(MEM_SIZE);
  ctx_.scratch_buffer.resize(SCRATCH_SIZE);
  ctx_.scratch = {0, ctx_.scratch_buffer.size(), ctx_.scratch_buffer.data()};
#ifdef GGML_USE_CUBLAS
  ggml_cuda_set_scratch_size(SCRATCH_SIZE);
#endif
  constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
  const size_t ctx_w_size = (3 + config.num_hidden_layers * (14 + 3 * config.num_experts)) * tensor_ovhd;
  const size_t ctx_kv_size = 2 * config.num_hidden_layers *
                             (config.max_length * config.hidden_size / config.num_attention_heads * config.num_kv_heads * ggml_type_size(GGML_TYPE_F16) + tensor_ovhd);
  ctx_.dtype = config.dtype;
  ctx_.ctx_w = make_unique_ggml_context(ctx_w_size, nullptr, true);
  ctx_.ctx_kv = make_unique_ggml_context(ctx_kv_size + 1 * MB, nullptr, false); // 1MB extra for MPS

  transformer = QwenMoeModel(&ctx_, config);
  lm_head = Linear(&ctx_, config.hidden_size, config.vocab_size, false);
  QWEN_CHECK(ggml_used_mem(ctx_.ctx_w.get()) == ggml_get_mem_size(ctx_.ctx_w.get())) << "corrupted model weights";
  QWEN_CHECK(ggml_used_mem(ctx_.ctx_kv.get()) == ctx_kv_size) << "corrupted kv cache";

  // build state_dict
  state_dict_.reserve(3 + config.num_hidden_layers * (14 + 3 * config.num_experts));
  state_dict_.emplace_back("model.embed_tokens.weight", transformer.embed_tokens.weight);

  for (int i = 0; i < config.num_hidden_layers; i++) {
    std::string layer_prefix = "model.layers." + std::to_string(i) + '.';
    state_dict_.emplace_back(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);
    
    state_dict_.emplace_back(layer_prefix + "self_attn.q_proj.weight",
                             transformer.layers[i].attn.q_proj.weight);
    state_dict_.emplace_back(layer_prefix + "self_attn.q_proj.bias",
                             transformer.layers[i].attn.q_proj.bias);

    state_dict_.emplace_back(layer_prefix + "self_attn.k_proj.weight",
                             transformer.layers[i].attn.k_proj.weight);
    state_dict_.emplace_back(layer_prefix + "self_attn.k_proj.bias",
                             transformer.layers[i].attn.k_proj.bias);

    state_dict_.emplace_back(layer_prefix + "self_attn.v_proj.weight",
                             transformer.layers[i].attn.v_proj.weight);
    state_dict_.emplace_back(layer_prefix + "self_attn.v_proj.bias",
                             transformer.layers[i].attn.v_proj.bias);

    state_dict_.emplace_back(layer_prefix + "self_attn.o_proj.weight",
                             transformer.layers[i].attn.o_proj.weight);

    state_dict_.emplace_back(layer_prefix + "post_attention_layernorm.weight",
                             transformer.layers[i].post_attention_layernorm.weight);

    state_dict_.emplace_back(layer_prefix + "mlp.gate.weight", transformer.layers[i].mlp.gate.weight);

    for (int j = 0; j < config.num_experts; j++){
        std::string prefix = layer_prefix + "mlp.experts." + std::to_string(j) + '.';
        state_dict_.emplace_back(prefix+"gate_proj.weight", transformer.layers[i].mlp.experts[j].gate_proj.weight);
        state_dict_.emplace_back(prefix+"up_proj.weight", transformer.layers[i].mlp.experts[j].up_proj.weight);
        state_dict_.emplace_back(prefix+"down_proj.weight", transformer.layers[i].mlp.experts[j].down_proj.weight);
    }
    state_dict_.emplace_back(layer_prefix + "mlp.shared_expert.gate_proj.weight", transformer.layers[i].mlp.shared_expert.gate_proj.weight);
    state_dict_.emplace_back(layer_prefix + "mlp.shared_expert.up_proj.weight", transformer.layers[i].mlp.shared_expert.up_proj.weight);
    state_dict_.emplace_back(layer_prefix + "mlp.shared_expert.down_proj.weight", transformer.layers[i].mlp.shared_expert.down_proj.weight);
    state_dict_.emplace_back(layer_prefix + "mlp.shared_expert_gate.weight", transformer.layers[i].mlp.shared_expert_gate.weight);
  }
  state_dict_.emplace_back("model.norm.weight", transformer.norm.weight);
  state_dict_.emplace_back("lm_head.weight", lm_head.weight);
}

QwenForCausalLM::~QwenForCausalLM() {
  for (auto &item : state_dict_) {
    tensor_to_cpu(item.second);
  }

  for (auto &layer : transformer.layers) {
    tensor_to_cpu(layer.attn.k_cache);
    tensor_to_cpu(layer.attn.v_cache);
  }
}

QwenMoeForCausalLM::~QwenMoeForCausalLM() {
  for (auto &item : state_dict_) {
    tensor_to_cpu(item.second);
  }

  for (auto &layer : transformer.layers) {
    tensor_to_cpu(layer.attn.k_cache);
    tensor_to_cpu(layer.attn.v_cache);
  }
}

auto QwenForCausalLM::generate_next_token(
  const std::vector<int32_t> &input_ids,
  const GenerationConfig &gen_config,
  int n_past,
  int n_ctx
) -> int32_t {
  ctx_.ctx_b = make_unique_ggml_context(ctx_.compute_buffer.size(), ctx_.compute_buffer.data(), false);
  // ctx_.gf = ggml_new_graph(ctx_.ctx_b.get()); // dafault graph size didn't fit larger model like 32b, moe
  size_t GRAPH_SIZE = 4096 * 2;
  ctx_.gf = ggml_new_graph_custom(ctx_.ctx_b.get(), GRAPH_SIZE, false);

  int n_threads = gen_config.num_threads; // user defined
  if (n_threads <= 0) {
    n_threads = get_default_num_threads(); // default thread num
  }
  int curr_input_ids_size = input_ids.size() - n_past;
  if (curr_input_ids_size >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas()) {
    n_threads = 1; // use 1 thread if BLAS is enabled
  }

  ggml_tensor *curr_input_ids = ggml_new_tensor_1d(ctx_.ctx_b.get(), GGML_TYPE_I32, curr_input_ids_size);
  memcpy(curr_input_ids->data, input_ids.data() + n_past, ggml_nbytes(curr_input_ids));

  ggml_tensor *KQ_pos = ggml_new_tensor_1d(ctx_.ctx_b.get(), GGML_TYPE_I32, curr_input_ids_size);
  int * data = static_cast<int *>(KQ_pos->data);
  for (int i = 0; i < curr_input_ids_size; ++i) {
    data[i] = n_past + i;
  }
  if (KQ_pos) {
    tensor_to_device(KQ_pos);
  }

  ggml_tensor *lm_logits = forward(&ctx_, curr_input_ids, KQ_pos, n_ctx);
  lm_logits->backend = GGML_BACKEND_CPU; // newer ggml version
  if (KQ_pos) {
    tensor_to_cpu(KQ_pos);
  }

  ggml_build_forward_expand(ctx_.gf, lm_logits);
#ifdef GGML_USE_METAL
  ggml_metal_graph_compute(ctx_.ctx_metal.get(), ctx_.gf);
#else
  ggml_graph_compute_helper(ctx_.work_buffer, ctx_.gf, n_threads);
#endif

  int vocab_size = lm_logits->ne[0];
  float *next_token_logits = (float *)lm_logits->data;

  // logits pre-process
  if (gen_config.repetition_penalty != 1.f) {
    sampling_repetition_penalty(next_token_logits, next_token_logits + vocab_size, input_ids,
                                gen_config.repetition_penalty);
  }

  int next_token_id;
  if (gen_config.do_sample) {
    // temperature sampling
    if (gen_config.temperature > 0) {
      sampling_temperature(next_token_logits, next_token_logits + vocab_size, gen_config.temperature);
    }

    std::vector<TokenIdScore> token_scores(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
      token_scores[i] = TokenIdScore(i, next_token_logits[i]);
    }

    // top_k sampling
    if (0 < gen_config.top_k && gen_config.top_k < (int)token_scores.size()) {
      sampling_top_k(token_scores.data(), token_scores.data() + gen_config.top_k,
                     token_scores.data() + token_scores.size());
      token_scores.resize(gen_config.top_k);
    }

    // top_p sampling
    if (0.f < gen_config.top_p && gen_config.top_p < 1.f) {
      auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), gen_config.top_p);
      token_scores.resize(pos - token_scores.data());
    }

    // sample next token
    sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
    for (size_t i = 0; i < token_scores.size(); i++) {
      next_token_logits[i] = token_scores[i].score;
    }

    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());

    std::discrete_distribution<> dist(next_token_logits, next_token_logits + token_scores.size());
    next_token_id = token_scores[dist(gen)].id;
  } else {
    // greedy search
    next_token_id = std::max_element(next_token_logits, next_token_logits + vocab_size) - next_token_logits;
  }

  return next_token_id;
}

auto QwenForCausalLM::sampling_repetition_penalty(
  float *first, float *last, const std::vector<int> &input_ids, float penalty
) -> void {
  QWEN_CHECK(penalty > 0) << "penalty must be a positive float, but got " << penalty;
  std::unordered_set<int> unique_input_ids(input_ids.begin(), input_ids.end());
  for (int id : unique_input_ids) {
    QWEN_CHECK(first <= first + id && first + id < last) << "invalid input id " << id;
    if (first[id] > 0) {
      first[id] /= penalty;
    } else {
      first[id] *= penalty;
    }
  }
}

auto QwenForCausalLM::sampling_temperature(
  float *first, float *last, float temp
) -> void {
  float inv_temp = 1.f / temp;
  for (float *it = first; it != last; it++) {
    *it *= inv_temp;
  }
}

auto QwenForCausalLM::sampling_top_k(
  TokenIdScore *first, TokenIdScore *kth, TokenIdScore *last
) -> void {
  std::nth_element(first, kth, last, std::greater<TokenIdScore>());
}

auto QwenForCausalLM::sampling_top_p(
  TokenIdScore *first, TokenIdScore *last, float top_p
) -> TokenIdScore * {
  // fast top_p in expected O(n) time complexity
  sampling_softmax_inplace(first, last);

  while (first + 1 < last) {
    float pivot_score = (last - 1)->score; // use mid score?
    TokenIdScore *mid =
      std::partition(first, last - 1, [pivot_score](const TokenIdScore &x) { return x.score > pivot_score; });
    std::swap(*mid, *(last - 1));

    float prefix_sum =
      std::accumulate(first, mid, 0.f, [](float sum, const TokenIdScore &x) { return sum + x.score; });
    if (prefix_sum >= top_p) {
      last = mid;
    } else if (prefix_sum + mid->score < top_p) {
      first = mid + 1;
      top_p -= prefix_sum + mid->score;
    } else {
      return mid + 1;
    }
  }
  return last;
}

auto QwenForCausalLM::sampling_softmax_inplace(
  TokenIdScore *first, TokenIdScore *last
) -> void {
  float max_score = std::max_element(first, last)->score;
  float sum = 0.f;
  for (TokenIdScore *p = first; p != last; p++) {
    float s = std::exp(p->score - max_score);
    p->score = s;
    sum += s;
  }
  float inv_sum = 1.f / sum;
  for (TokenIdScore *p = first; p != last; p++) {
    p->score *= inv_sum;
  }
}

auto QwenForCausalLM::generate(
  const std::vector<int> &input_ids,
  const GenerationConfig &gen_config,
  BaseStreamer *streamer
) -> std::vector<int> {
  std::vector<int> output_ids;
  output_ids.reserve(gen_config.max_length);
  output_ids = input_ids;
  if (streamer) {
    streamer->put(input_ids);
  }

  int n_past = 0;
  const int n_ctx = input_ids.size();

  while ((int)output_ids.size() < gen_config.max_length) {
    auto next_token_id = generate_next_token(output_ids, gen_config, n_past, n_ctx);

    n_past = output_ids.size();
    output_ids.emplace_back(next_token_id);

    if (streamer) {
      streamer->put({next_token_id});
    }

    if (next_token_id == config.eos_token_id || next_token_id == config.im_start_id || next_token_id == config.im_end_id) {
      break;
    }
  }

  if (streamer) {
    streamer->end();
  }

  return output_ids;
}

auto QwenForCausalLM::load(ModelLoader &loader) -> void {
  // std::cout << "Debug: state_dict size: " << state_dict_.size() <<std::endl;
  for (auto &item : state_dict_) {
    const std::string &name = item.first;
    ggml_tensor *tensor = item.second;
    loader.read_tensor(name, tensor);
    if (name != "model.embed_tokens.weight") {
      tensor_to_device(tensor);
    }
  }

  for (auto &layer : transformer.layers) {
    tensor_to_device(layer.attn.k_cache);
    tensor_to_device(layer.attn.v_cache);
  }

  ctx_.weight_buffer = std::string_view(loader.data, loader.size);
  ctx_.init_device_context();
}

auto QwenMoeForCausalLM::load(ModelLoader &loader) -> void {
  for (auto &item : state_dict_) {
    const std::string &name = item.first;
    ggml_tensor *tensor = item.second;
    loader.read_tensor(name, tensor);
    if (name != "model.embed_tokens.weight") {
      tensor_to_device(tensor);
    }
  }

  for (auto &layer : transformer.layers) {
    tensor_to_device(layer.attn.k_cache);
    tensor_to_device(layer.attn.v_cache);
  }

  ctx_.weight_buffer = std::string_view(loader.data, loader.size);
  ctx_.init_device_context();
}

auto QwenForCausalLM::forward(
  ModelContext *ctx,
  ggml_tensor *input_ids,
  ggml_tensor *KQ_pos,
  int n_ctx
) const -> ggml_tensor * {
  ggml_tensor *transformer_outputs = transformer.forward(ctx, input_ids, KQ_pos, n_ctx);
  // NOTE: only compute next_token_logits for the last token
  if (input_ids->ne[0] > 1) {
    transformer_outputs = tensor_assign_buffers(
      ggml_view_1d(ctx->ctx_b.get(), transformer_outputs, config.hidden_size,
                   (input_ids->ne[0] - 1) * config.hidden_size * ggml_element_size(transformer_outputs)));
  }
  ggml_tensor *lm_logits = lm_head.forward(ctx, transformer_outputs);
  return lm_logits;
}

auto QwenMoeForCausalLM::forward(
  ModelContext *ctx,
  ggml_tensor *input_ids,
  ggml_tensor *KQ_pos,
  int n_ctx
) const -> ggml_tensor * {
  ggml_tensor *transformer_outputs = transformer.forward(ctx, input_ids, KQ_pos, n_ctx, config.num_experts, config.num_experts_per_tok);
  // NOTE: only compute next_token_logits for the last token
  if (input_ids->ne[0] > 1) {
    transformer_outputs = tensor_assign_buffers(
      ggml_view_1d(ctx->ctx_b.get(), transformer_outputs, config.hidden_size,
                   (input_ids->ne[0] - 1) * config.hidden_size * ggml_element_size(transformer_outputs)));
  }
  ggml_tensor *lm_logits = lm_head.forward(ctx, transformer_outputs);
  return lm_logits;
}

// ===== pipeline =====

Pipeline::Pipeline(const std::string &path, const std::string &tiktoken_path, int max_length) {
  auto _update_config_max_length = [](QwenConfig &config, int max_length) {
      if (max_length > 0) {
          QWEN_CHECK(max_length <= config.max_length)
              << "Requested max_length (" << max_length << ") exceeds the max possible model sequence length ("
              << config.max_length;
          config.max_length = max_length;
      }
  };
  mapped_file = std::make_unique<MappedFile>(path);
  ModelLoader loader(std::string_view((char *)mapped_file->data, mapped_file->size));

  // load magic
  std::string magic = loader.read_string(4);
  QWEN_CHECK(magic == "ggml") << "model file is broken (bad magic)";

  // load model type
  ModelType model_type = (ModelType)loader.read_basic<int>();
  // load version
  int version = loader.read_basic<int>();

  std::unique_ptr<QwenConfig> config;
  if (model_type == ModelType::QWEN2) {
      // load config
      config = std::make_unique<QwenConfig>(loader.read_basic<QwenConfig>());
      _update_config_max_length(*config, max_length);
      model = std::make_unique<QwenForCausalLM>(*config);
  }
  else if (model_type == ModelType::QWEN2MOE)
  {
      config = std::make_unique<QwenMoeConfig>(loader.read_basic<QwenMoeConfig>());
      _update_config_max_length(*config, max_length);
      QwenMoeConfig* moe_config = static_cast<QwenMoeConfig*>(config.get());
      model = std::make_unique<QwenMoeForCausalLM>(*moe_config);
  }

  // load model
  model->load(loader);
  // std::cout<< "weight load complish"<<std::endl;

  // load tokenizer
  tokenizer = std::make_unique<QwenTokenizer>(tiktoken_path, *config);
}

auto Pipeline::generate(
  const std::vector<int> &input_ids,
  const GenerationConfig &gen_config,
  BaseStreamer *streamer
) const -> std::vector<int> {
  std::vector<int> output_ids = model->generate(input_ids, gen_config, streamer);
  std::vector<int> new_output_ids(output_ids.begin() + input_ids.size(), output_ids.end());
  return new_output_ids;
}

auto Pipeline::generate(
  const std::string &prompt,
  const GenerationConfig &gen_config,
  BaseStreamer *streamer
) const -> std::string {
  std::vector<int> input_ids = tokenizer->encode(prompt, gen_config.max_context_length);
  std::vector<int> output_ids = model->generate(input_ids, gen_config, streamer);

  std::vector<int> new_output_ids(output_ids.begin() + input_ids.size(), output_ids.end());
  std::string output = tokenizer->decode(new_output_ids);
  return output;
}

auto Pipeline::chat(const std::vector<ChatMessage> &messages, const GenerationConfig &gen_config,
                    BaseStreamer *streamer) const -> ChatMessage {
  std::vector<int> input_ids = tokenizer->encode_messages(messages, gen_config.max_context_length);
  std::vector<int> new_output_ids = generate(input_ids, gen_config, streamer);
  ChatMessage output = tokenizer->decode_message(new_output_ids);
  return output;
}

} // namespace qwen
