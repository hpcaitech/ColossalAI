#pragma once

#include <memory>

#include "common/nvgpu_dev_info.h"
#include "target.h"

namespace colossalAI {
namespace common {

template <typename Ret>
class DevInfoMgr final {
 public:
  static std::unique_ptr<Ret> GetDevInfo(int device_num) const {
    return std::make_unique<Ret>(device_num);
  }
};

}  // namespace common
}  // namespace colossalAI
