#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

const int32_t LONG_SENTENCE_LEN = 512;

struct MaskedLMInstance {
  int index;
  std::string label;
  MaskedLMInstance(int index, std::string label) {
    this->index = index;
    this->label = label;
  }
};

auto get_new_segment(
    std::vector<std::string> segment, std::vector<std::string> segment_jieba,
    const std::vector<bool> chinese_vocab) {  // const
                                              // std::unordered_set<std::string>
                                              // &chinese_vocab
  std::unordered_set<std::string> seq_cws_dict;
  for (auto word : segment_jieba) {
    seq_cws_dict.insert(word);
  }
  int i = 0;
  std::vector<std::string> new_segment;
  int segment_size = segment.size();
  while (i < segment_size) {
    if (!chinese_vocab[i]) {  // chinese_vocab.find(segment[i]) ==
                              // chinese_vocab.end()
      new_segment.emplace_back(segment[i]);
      i += 1;
      continue;
    }
    bool has_add = false;
    for (int length = 3; length >= 1; length--) {
      if (i + length > segment_size) {
        continue;
      }
      std::string chinese_word = "";
      for (int j = i; j < i + length; j++) {
        chinese_word += segment[j];
      }
      if (seq_cws_dict.find(chinese_word) != seq_cws_dict.end()) {
        new_segment.emplace_back(segment[i]);
        for (int j = i + 1; j < i + length; j++) {
          new_segment.emplace_back("##" + segment[j]);
        }
        i += length;
        has_add = true;
        break;
      }
    }
    if (!has_add) {
      new_segment.emplace_back(segment[i]);
      i += 1;
    }
  }

  return new_segment;
}

bool startsWith(const std::string &s, const std::string &sub) {
  return s.find(sub) == 0 ? true : false;
}

auto create_whole_masked_lm_predictions(
    std::vector<std::string> &tokens,
    const std::vector<std::string> &original_tokens,
    const std::vector<std::string> &vocab_words,
    std::map<std::string, int> &vocab, const int max_predictions_per_seq,
    const double masked_lm_prob) {
  // for (auto item : vocab) {
  //     std::cout << "key=" << std::string(py::str(item.first)) << ", "
  //               << "value=" << std::string(py::str(item.second)) <<
  //               std::endl;
  // }
  std::vector<std::vector<int> > cand_indexes;
  std::vector<int> cand_temp;
  int tokens_size = tokens.size();
  std::string prefix = "##";
  bool do_whole_masked = true;

  for (int i = 0; i < tokens_size; i++) {
    if (tokens[i] == "[CLS]" || tokens[i] == "[SEP]") {
      continue;
    }
    if (do_whole_masked && (cand_indexes.size() > 0) &&
        (tokens[i].rfind(prefix, 0) == 0)) {
      cand_temp.emplace_back(i);
    } else {
      if (cand_temp.size() > 0) {
        cand_indexes.emplace_back(cand_temp);
      }
      cand_temp.clear();
      cand_temp.emplace_back(i);
    }
  }
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(cand_indexes.begin(), cand_indexes.end(),
               std::default_random_engine(seed));
  // for (auto i : cand_indexes) {
  //     for (auto j : i) {
  //         std::cout << tokens[j] << " ";
  //     }
  //     std::cout << std::endl;
  // }
  // for (auto i : output_tokens) {
  //     std::cout << i;
  // }
  // std::cout << std::endl;

  int num_to_predict = std::min(max_predictions_per_seq,
                                std::max(1, int(tokens_size * masked_lm_prob)));
  // std::cout << num_to_predict << std::endl;

  std::set<int> covered_indexes;
  std::vector<int> masked_lm_output(tokens_size, -1);
  int vocab_words_len = vocab_words.size();
  std::default_random_engine e(seed);
  std::uniform_real_distribution<double> u1(0.0, 1.0);
  std::uniform_int_distribution<unsigned> u2(0, vocab_words_len - 1);
  int mask_cnt = 0;
  std::vector<std::string> output_tokens;
  output_tokens = original_tokens;

  for (auto index_set : cand_indexes) {
    if (mask_cnt > num_to_predict) {
      break;
    }
    int index_set_size = index_set.size();
    if (mask_cnt + index_set_size > num_to_predict) {
      continue;
    }
    bool is_any_index_covered = false;
    for (auto index : index_set) {
      if (covered_indexes.find(index) != covered_indexes.end()) {
        is_any_index_covered = true;
        break;
      }
    }
    if (is_any_index_covered) {
      continue;
    }
    for (auto index : index_set) {
      covered_indexes.insert(index);
      std::string masked_token;
      if (u1(e) < 0.8) {
        masked_token = "[MASK]";
      } else {
        if (u1(e) < 0.5) {
          masked_token = output_tokens[index];
        } else {
          int random_index = u2(e);
          masked_token = vocab_words[random_index];
        }
      }
      // masked_lms.emplace_back(MaskedLMInstance(index, output_tokens[index]));
      masked_lm_output[index] = vocab[output_tokens[index]];
      output_tokens[index] = masked_token;
      mask_cnt++;
    }
  }

  // for (auto p : masked_lms) {
  //     masked_lm_output[p.index] = vocab[p.label];
  // }
  return std::make_tuple(output_tokens, masked_lm_output);
}

PYBIND11_MODULE(mask, m) {
  m.def("create_whole_masked_lm_predictions",
        &create_whole_masked_lm_predictions);
  m.def("get_new_segment", &get_new_segment);
}
