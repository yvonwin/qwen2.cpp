#pragma once

#include "tiktoken.h"

#include <ggml.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#endif

#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif

namespace qwen {

class QwenTokenizer;

// ===== common =====

static constexpr size_t MB = 1024 * 1024;

static const std::string PAT_STR = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

class LogMessageFatal {
  public:
    LogMessageFatal(const char *file, int line) { oss_ << file << ':' << line << ' '; }
    [[noreturn]] ~LogMessageFatal() noexcept(false) { throw std::runtime_error(oss_.str()); }
    auto stream() -> std::ostringstream & { return oss_; }

  private:
    std::ostringstream oss_;
};

#define QWEN_THROW ::qwen::LogMessageFatal(__FILE__, __LINE__).stream()
#define QWEN_CHECK(cond) \
    if (!(cond)) \
    QWEN_THROW << "check failed (" #cond ") "

ggml_tensor *tensor_assign_buffers(ggml_tensor *tensor);

auto tensor_to_device(ggml_tensor *tensor) -> ggml_tensor *;

auto tensor_to_cpu(ggml_tensor *tensor) -> ggml_tensor *;

auto get_num_physical_cores() -> int;
auto get_default_num_threads() -> int;

struct ggml_context_deleter_t {
  auto operator()(ggml_context *ctx) const noexcept -> void { ggml_free(ctx); }
};

using unique_ggml_context_t = std::unique_ptr<ggml_context, ggml_context_deleter_t>;

static inline auto make_unique_ggml_context(
  size_t mem_size, void *mem_buffer, bool no_alloc
) -> unique_ggml_context_t {
  return unique_ggml_context_t(ggml_init({mem_size, mem_buffer, no_alloc}));
}

#ifdef GGML_USE_METAL
struct ggml_metal_context_deleter_t {
  auto operator()(ggml_metal_context *ctx) const noexcept -> void { ggml_metal_free(ctx); }
};

using unique_ggml_metal_context_t = std::unique_ptr<ggml_metal_context, ggml_metal_context_deleter_t>;

static inline auto make_unique_ggml_metal_context(
  int n_cb
) -> unique_ggml_metal_context_t {
  return unique_ggml_metal_context_t(ggml_metal_init(n_cb));
}
#endif

struct uninitialized_char {
  char m;
  uninitialized_char() {}
};

auto ggml_graph_compute_helper(std::vector<uninitialized_char> &buf, ggml_cgraph *graph, int n_threads) -> void;

struct ModelContext {
  ggml_type dtype;
  unique_ggml_context_t ctx_w;  // weight
  unique_ggml_context_t ctx_kv; // kv cache
  unique_ggml_context_t ctx_b;  // buffer
#ifdef GGML_USE_METAL
  unique_ggml_metal_context_t ctx_metal;
#endif
  ggml_cgraph *gf;
  ggml_scratch scratch;
  std::vector<uninitialized_char> compute_buffer; // BLAS buffer
  std::vector<uninitialized_char> scratch_buffer; // intermediate tensor buffer
  std::string_view weight_buffer;                 // mapped weight
  std::vector<uninitialized_char> work_buffer;    // temporary buffer for graph computing

  auto init_device_context() -> void;
};

class Embedding {
  public:
    Embedding() : weight(nullptr) {}
    Embedding(ModelContext *ctx, int num_embeddings, int embedding_dim)
      : weight(ggml_new_tensor_2d(ctx->ctx_w.get(), ctx->dtype, embedding_dim, num_embeddings)) {}

    auto forward(ModelContext *ctx, ggml_tensor *input) const -> ggml_tensor *;

    ggml_tensor *weight;
};

class Linear {
  public:
    Linear() : weight(nullptr), bias(nullptr) {}
    Linear(ModelContext *ctx, int in_features, int out_features, bool use_bias = true)
      : weight(ggml_new_tensor_2d(ctx->ctx_w.get(), ctx->dtype, in_features, out_features)),
        bias(use_bias ? ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, out_features) : nullptr) {}

    auto in_features() const -> int { return weight->ne[0]; }
    auto out_features() const -> int { return weight->ne[1]; }

    auto forward(ModelContext *ctx, ggml_tensor *input) const -> ggml_tensor *;

    ggml_tensor *weight; // [out_features, in_features]
    ggml_tensor *bias;   // [out_features]
};

class RMSNorm {
  public:
    RMSNorm() : weight(nullptr), inplace(true) {}
    RMSNorm(ModelContext *ctx, int normalized_shape, bool inplace = true)
      : weight(ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)), inplace(inplace) {}

    auto forward(ModelContext *ctx, ggml_tensor *input, float eps = 1e-5f) const -> ggml_tensor *;

    ggml_tensor *weight;
    bool inplace;
};

class BaseStreamer{
  public:
    virtual ~BaseStreamer() = default;
    virtual auto put(const std::vector<int> &output_ids) -> void = 0;
    virtual auto end() -> void = 0;
};

class StreamerGroup : public BaseStreamer {
  public:
    StreamerGroup(std::vector<std::shared_ptr<BaseStreamer>> streamers) : streamers_(std::move(streamers)) {}
    auto put(const std::vector<int> &output_ids) -> void override;
    auto end() -> void override;

  private:
    std::vector<std::shared_ptr<BaseStreamer>> streamers_;
};

// reference: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
class TextStreamer : public BaseStreamer {
  public:
    TextStreamer(std::ostream &os, QwenTokenizer *tokenizer)
        : os_(os), tokenizer_(tokenizer), is_prompt_(true), print_len_(0) {}
    auto put(const std::vector<int> &output_ids) -> void override;
    auto end() -> void override;

  private:
    std::ostream &os_;
    QwenTokenizer *tokenizer_;
    bool is_prompt_;
    std::vector<int> token_cache_;
    int print_len_;
};

class PerfStreamer : public BaseStreamer {
  public:
    PerfStreamer() : start_us_(0), prompt_us_(0), end_us_(0), num_prompt_tokens_(0), num_output_tokens_(0) {}

    auto put(const std::vector<int> &output_ids) -> void override;
    auto end() -> void override { end_us_ = ggml_time_us(); }

    auto reset() -> void;
    auto to_string() -> std::string const;

    auto num_prompt_tokens() const -> int64_t { return num_prompt_tokens_; }
    auto prompt_total_time_us() const -> int64_t { return prompt_us_ - start_us_; }
    auto prompt_token_time_us() const -> int64_t {
      return num_prompt_tokens() ? prompt_total_time_us() / num_prompt_tokens() : 0;
    }
    auto num_output_tokens() const -> int64_t { return num_output_tokens_; }
    auto output_total_time_us() const -> int64_t { return end_us_ - prompt_us_; }
    auto output_token_time_us() const -> int64_t {
      return num_output_tokens() ? output_total_time_us() / num_output_tokens() : 0;
    }

  private:
    int64_t start_us_;
    int64_t prompt_us_;
    int64_t end_us_;
    int64_t num_prompt_tokens_;
    int64_t num_output_tokens_;
};

class MappedFile {
  public:
    MappedFile(const std::string &path);
    ~MappedFile();

  public:
    char *data;
    size_t size;
};

class ModelLoader {
  public:
    ModelLoader(std::string_view buffer) : data(buffer.data()), size(buffer.size()), ptr(buffer.data()) {}

    auto tell() const -> int64_t { return ptr - data; }

    auto seek(int64_t offset, int whence) -> void;

    template <typename T>
    auto read_basic() -> T {
      T obj = *(T *)ptr;
      ptr += sizeof(T);
      return obj;
    }

    auto read_string(size_t length) -> std::string;

    auto read_tensor(const std::string &name, ggml_tensor *tensor) -> void;

  public:
    const char *const data;
    size_t size;
    const char *ptr;
};

// ===== generation =====

struct GenerationConfig {
  int max_length;
  int max_context_length;
  bool do_sample;
  int top_k;
  float top_p;
  float temperature;
  float repetition_penalty;
  int num_threads;

  GenerationConfig(int max_length = 2048, int max_context_length = 512, bool do_sample = true, int top_k = 0,
                   float top_p = 0.7, float temperature = 0.95, float repetition_penalty = 1.f, int num_threads = 0)
      : max_length(max_length), max_context_length(max_context_length), do_sample(do_sample), top_k(top_k),
        top_p(top_p), temperature(temperature), repetition_penalty(repetition_penalty), num_threads(num_threads) {}
};

struct TokenIdScore {
  int id;
  float score;

  TokenIdScore() = default;
  TokenIdScore(int id, float score) : id(id), score(score) {}

  auto operator<(const TokenIdScore &other) const -> bool { return score < other.score; }
  auto operator>(const TokenIdScore &other) const -> bool { return score > other.score; }

  friend auto operator<<(std::ostream &os, const TokenIdScore &self) -> std::ostream & {
    return os << "TokenIdScore(id=" << self.id << ", score=" << self.score << ")";
  }
};

// ===== Qwen1.5 =====

enum class ModelType {
    QWEN1 = 1,
    QWEN2 = 2,
    QWEN2MOE = 3,
};

struct QwenConfig {
  // common attributes
  ggml_type dtype;
  int vocab_size;
  int hidden_size;
  int num_attention_heads;
  int num_kv_heads;
  int num_hidden_layers;
  int intermediate_size;
  // for sequence generation
  int max_length;
  // for tokenizer
  int eos_token_id;
  int pad_token_id;
  int im_start_id;
  int im_end_id;
};

struct QwenMoeConfig : QwenConfig {
  int moe_intermediate_size;
  int shared_expert_intermediate_size;
  int num_experts;
  int num_experts_per_tok;
  int norm_topk_prob;
};

struct ChatMessage{
  std::string role;
  std::string content;

  static const std::string ROLE_USER;
  static const std::string ROLE_ASSISTANT;
  static const std::string ROLE_SYSTEM;

  ChatMessage() = default;
  ChatMessage(std::string role, std::string content)
      : role(std::move(role)), content(std::move(content)) {}

  friend std::ostream &operator<<(std::ostream &os, const ChatMessage &self) {
      os << "ChatMessage(role=" << std::quoted(self.role) << ", content=" << std::quoted(self.content);
      return os << ")";
  }
};

class QwenTokenizer {
  public:

    QwenTokenizer(const std::string & tiktoken_path, const QwenConfig &config);

    auto encode(const std::string &text, int max_length) const -> std::vector<int>;

    auto decode(const std::vector<int> &ids) const -> std::string;  

    std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const;

    ChatMessage decode_message(const std::vector<int> &ids) const{
       return {ChatMessage::ROLE_ASSISTANT, decode(ids)};
    };

    static std::string build_prompt(const std::vector<ChatMessage> &messages);

    auto is_special_id(int id) const -> bool;

    tiktoken::tiktoken tokenizer;
    int eos_token_id;
    int im_start_id;
    int im_end_id;

  protected:
    static void check_chat_messages(const std::vector<ChatMessage> &messages);
};

class QwenAttention {
  public:
    QwenAttention() : num_attention_heads(0), num_kv_heads(0) {}
    QwenAttention(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length);

    auto forward(ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor *;

    int num_attention_heads;
    int num_kv_heads;
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear o_proj;
    ggml_tensor *k_cache; // [n_head, maxlen, head_size]
    ggml_tensor *v_cache; // [n_head, head_size, maxlen]
};

class QwenMLP {
  public:
    QwenMLP() = default;
    QwenMLP(ModelContext * ctx, int hidden_size, int intermediate_size)
      : gate_proj(ctx, hidden_size, intermediate_size, false), // w2
        up_proj(ctx, hidden_size, intermediate_size, false),  // w1
        down_proj(ctx, intermediate_size, hidden_size, false) {}

    auto forward(ModelContext *ctx, ggml_tensor *hidden_states) const -> ggml_tensor *;

    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;
};

class Qwen2MoeSparseMoeBlock{
  public:
      Qwen2MoeSparseMoeBlock() = default;
      Qwen2MoeSparseMoeBlock(ModelContext * ctx, int hidden_size, int intermediate_size, int moe_intermediate_size, int shared_expert_intermediate_size, int num_experts, int num_experts_per_tok)
        :
        gate(ctx, hidden_size, num_experts, false),
        shared_expert(ctx, hidden_size, shared_expert_intermediate_size),
        shared_expert_gate(ctx, hidden_size, 1, false),
        norm_topk_prob(false)

      {
          for (int i = 0; i < num_experts; i++)
          {
              experts.emplace_back(QwenMLP(ctx, hidden_size, moe_intermediate_size));
              expert_gates.push_back(experts[i].gate_proj.weight);
              expert_downs.push_back(experts[i].down_proj.weight);
              expert_ups.push_back(experts[i].up_proj.weight);
          }
      }

      auto forward(ModelContext *ctx, ggml_tensor *hidden_states, int num_experts, int num_experts_per_tok) const -> ggml_tensor *;

      Linear gate;
      std::vector<QwenMLP> experts;
      std::vector<ggml_tensor *> expert_gates;
      std::vector<ggml_tensor *> expert_ups;
      std::vector<ggml_tensor *> expert_downs;
      QwenMLP shared_expert;
      Linear shared_expert_gate;
      bool norm_topk_prob;
};


class QwenBlock {
  public:
    QwenBlock() = default;
    QwenBlock(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int intermediate_size, int max_length)
      : input_layernorm(ctx, hidden_size, false),
        attn(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length),
        post_attention_layernorm(ctx, hidden_size, false),
        mlp(ctx, hidden_size, intermediate_size) {}

    auto forward(ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor *;

    RMSNorm input_layernorm;
    QwenAttention attn;
    RMSNorm post_attention_layernorm;
    QwenMLP mlp;
};

class QwenMoeBlock {
  public:
    QwenMoeBlock() = default;
    QwenMoeBlock(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int intermediate_size, int moe_intermediate_size, int shared_expert_intermediate_size, int num_experts, int num_experts_per_tok, int max_length)
      : input_layernorm(ctx, hidden_size, false),
        attn(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length),
        post_attention_layernorm(ctx, hidden_size, false),
        mlp(ctx, hidden_size, intermediate_size, moe_intermediate_size, shared_expert_intermediate_size, num_experts,num_experts_per_tok) {}

    auto forward(ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *KQ_pos, int n_ctx, int num_experts, int num_experts_per_tok) const -> ggml_tensor *;

    RMSNorm input_layernorm;
    QwenAttention attn;
    RMSNorm post_attention_layernorm;
    Qwen2MoeSparseMoeBlock mlp;
};

class QwenModel {
  public:
    QwenModel() = default;
    QwenModel(ModelContext *ctx, const QwenConfig &config);

    auto forward(ModelContext *ctx, ggml_tensor *input_ids, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor *;

    Embedding embed_tokens;
    std::vector<QwenBlock> layers;
    RMSNorm norm;
};

class QwenMoeModel {
  public:
    QwenMoeModel() = default;
    QwenMoeModel(ModelContext *ctx, const QwenMoeConfig &config);

    // Attention: These parameters should not be set to fixed values. I did this for quick implementation.
    auto forward(ModelContext *ctx, ggml_tensor *input_ids, ggml_tensor *KQ_pos, int n_ctx, int num_experts=60, int num_experts_per_tok=4) const -> ggml_tensor *;

    Embedding embed_tokens;
    std::vector<QwenMoeBlock> layers;
    RMSNorm norm;
};

class QwenForCausalLM {
  public:
    QwenForCausalLM(const QwenConfig &config);
    ~QwenForCausalLM();

    auto generate_next_token(
      const std::vector<int> &input_ids,
      const GenerationConfig &gen_config,
      int n_past,
      int n_ctx
    ) -> int;

    auto generate(
      const std::vector<int> &input_ids,
      const GenerationConfig &gen_config,
      BaseStreamer *streamer = nullptr
    ) -> std::vector<int>;

    // logits processor
    static auto sampling_repetition_penalty(float *first, float *last, const std::vector<int32_t> &input_ids,
                                            float penalty) -> void;
    // logits warper
    static auto sampling_temperature(float *first, float *last, float temp) -> void;
    static auto sampling_top_k(TokenIdScore *first, TokenIdScore *kth, TokenIdScore *last) -> void;
    static auto sampling_top_p(TokenIdScore *first, TokenIdScore *last, float top_p) -> TokenIdScore *;

    static auto sampling_softmax_inplace(TokenIdScore *first, TokenIdScore *last) -> void;

    virtual void load(ModelLoader &loader);

    virtual ggml_tensor * forward(ModelContext *ctx, ggml_tensor *input_ids, ggml_tensor *KQ_pos, int n_ctx) const;

    static constexpr size_t MEM_SIZE     = 1280 * MB;  // 2k context
    static constexpr size_t SCRATCH_SIZE = 1280 * MB; // 2k context

    QwenConfig config;
    QwenModel transformer;
    Linear lm_head;

    private:
      ModelContext ctx_;
      std::vector<std::pair<std::string, ggml_tensor *>> state_dict_;
};

class QwenMoeForCausalLM : public QwenForCausalLM {
  public:
    QwenMoeForCausalLM(const QwenMoeConfig &config);  // Declaration
    ~QwenMoeForCausalLM();
    // Override methods here if needed

    auto load(ModelLoader &loader) -> void override;
    auto forward(ModelContext *ctx, ggml_tensor *input_ids, ggml_tensor *KQ_pos, int n_ctx) const -> ggml_tensor * override;

    static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 1844ull * 1024 * 1024;
    QwenMoeConfig config;
    QwenMoeModel transformer;
  
  private:
    ModelContext ctx_;
    std::vector<std::pair<std::string, ggml_tensor *>> state_dict_;
};

// ===== pipeline =====

class Pipeline {
  public:
    Pipeline(const std::string &path, const std::string &tiktoken_path, int max_length = -1);

    auto generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                  BaseStreamer *streamer = nullptr) const -> std::vector<int>;

    auto generate(const std::string &prompt, const GenerationConfig &gen_config,
                  BaseStreamer *streamer = nullptr) const -> std::string;

    auto chat(const std::vector<ChatMessage> &messages, const GenerationConfig &gen_config,
              BaseStreamer *streamer = nullptr) const -> ChatMessage;

  public:
    std::unique_ptr<QwenTokenizer> tokenizer;
    std::unique_ptr<QwenForCausalLM> model;
    std::unique_ptr<MappedFile> mapped_file;
};

} // namespace qwen
