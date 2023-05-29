#include <Python.h>
#include <bits/stdc++.h>
#include <torch/extension.h>

int move_count_impl(std::vector<std::vector<int>> &steps, int n_blocks) {
  int n_steps = steps.size();
  std::unordered_map<int, int> my_map;
  std::map<std::pair<int, int>, int> next_map;

  for (auto i = n_steps - 1; ~i; --i) {
    auto ids = steps.at(i);
    for (auto c_id : ids) {
      auto iter = my_map.find(c_id);
      auto nxt = n_steps;
      if (iter != my_map.end()) nxt = iter->second;
      next_map.emplace(std::make_pair(i, c_id), nxt);
      my_map[c_id] = i;
    }
  }
  // reuse this map
  for (auto iter : my_map) my_map[iter.first] = 0;

  int cache_size = 0, count = 0;
  std::priority_queue<std::pair<int, int>> cache;
  for (auto i = 0; i < n_steps; ++i) {
    auto ids = steps.at(i);
    assert(n_blocks >= ids.size());

    int not_in = 0;
    for (auto c_id : ids)
      if (my_map[c_id] == 0) ++not_in;

    while (cache_size + not_in > n_blocks) {
      std::pair<int, int> q_top = cache.top();
      cache.pop();
      assert(q_top.first > i);
      assert(my_map[q_top.second] == 1);
      my_map[q_top.second] = 0;
      --cache_size;
      ++count;
    }

    for (auto c_id : ids) {
      auto iter = next_map.find(std::make_pair(i, c_id));
      cache.push(std::make_pair(iter->second, c_id));
      if (my_map[c_id] == 0) {
        my_map[c_id] = 1;
        ++cache_size;
      }
    }
  }
  return (count + cache_size) << 1;
}

int move_count(std::vector<std::vector<int>> &steps, int n_blocks) {
  return move_count_impl(steps, n_blocks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("move_count", &move_count, "Count the number of moves.");
}
