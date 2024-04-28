#pragma once

#include <re2/re2.h>
#include <cassert>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tiktoken {

class tiktoken {
public:
    tiktoken();
    tiktoken(
        std::unordered_map<std::string, int> encoder,
        std::unordered_map<std::string, int> special_encoder,
        const std::string &pattern
    );

    auto encode_ordinary(const std::string &text) const -> std::vector<int>;
    auto encode(const std::string &text) const -> std::vector<int>;
    auto encode_single_piece(const std::string &text) const -> std::vector<int>;
    auto decode(const std::vector<int> &tokens) const -> std::string;

private:
    auto split_with_allowed_special_token(
        re2::StringPiece &input,
        const std::unordered_map<std::string, int> &allowed_special
    ) const -> std::pair<std::optional<std::string>, re2::StringPiece>;

    auto _encode_ordinary_native(const std::string &text) const -> std::vector<int>;
    auto _encode_native(
        const std::string &text,
        const std::unordered_map<std::string, int> &allowed_special
    ) const -> std::pair<std::vector<int>, int>;
    auto _decode_native(const std::vector<int> &tokens) const -> std::string;

    std::unordered_map<std::string, int> encoder_;
    std::unordered_map<std::string, int> special_tokens_encoder;
    std::unordered_map<int, std::string> decoder_;
    std::unordered_map<int, std::string> special_tokens_decoder;
    std::unique_ptr<re2::RE2> regex_;
    std::unique_ptr<re2::RE2> special_regex_;
};

} // namespace tiktoken