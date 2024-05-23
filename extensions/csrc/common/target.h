#pragma once

#include <exception>
#include <iostream>
#include <string>

namespace colossalAI {
namespace common {

class Target {
 public:
  enum class OS : int {
    Unk = -1,
    Linux,
    Windows,
  };
  enum class Arch : int {
    Unk = -1,
    X86,
    Arm,
    NVGPU,
    AMDGPU,
    Ascend,
  };
  enum class BitLen : int {
    Unk = -1,
    k32,
    k64,
  };

  explicit Target(OS os, Arch arch, BitLen bitlen)
      : os_(os), arch_(arch), bitlen_(bitlen) {}

  bool defined() const {
    return (os_ != OS::Unk) && (arch_ != Arch::Unk) && (bitlen_ != BitLen::Unk);
  }

  std::string str() const {
    std::string s{"OS: "};
    switch (os_) {
      case OS::Unk:
        s += "Unk";
        break;
      case OS::Linux:
        s += "Linux";
        break;
      case OS::Windows:
        s += "Windows";
        break;
      default:
        throw std::invalid_argument("Invalid OS type!");
    }
    s += "\t";
    s += "Arch: ";

    switch (arch_) {
      case Arch::Unk:
        s += "Unk";
        break;
      case Arch::X86:
        s += "X86";
        break;
      case Arch::Arm:
        s += "Arm";
        break;
      case Arch::NVGPU:
        s += "NVGPU";
        break;
      case Arch::AMDGPU:
        s += "AMDGPU";
        break;
      case Arch::Ascend:
        s += "Ascend";
        break;
      default:
        throw std::invalid_argument("Invalid Arch type!");
    }
    s += "\t";
    s += "BitLen: ";

    switch (bitlen_) {
      case BitLen::Unk:
        s += "Unk";
        break;
      case BitLen::k32:
        s += "k32";
        break;
      case BitLen::k64:
        s += "k64";
        break;
      default:
        throw std::invalid_argument("Invalid target bit length!");
    }

    return s;
  }

  OS os() const { return os_; }
  Arch arch() const { return arch_; }
  BitLen bitlen() const { return bitlen_; }

  static Target DefaultX86Target();
  static Target DefaultArmTarget();
  static Target DefaultRocmTarget();
  static Target DefaultAscendTarget();

  static Target DefaultCUDATarget() {
    return Target(OS::Linux, Arch::NVGPU, BitLen::k64);
  }

  friend std::ostream& operator<<(std::ostream& os, const Target& target);
  friend bool operator==(const Target& lhs, const Target& rhs);
  friend bool operator!=(const Target& lhs, const Target& rhs);

 private:
  OS os_{OS::Unk};
  Arch arch_{Arch::Unk};
  BitLen bitlen_{BitLen::Unk};
};

std::ostream& operator<<(std::ostream& os, const Target& target) {
  std::cout << target.str() << std::endl;
}
bool operator==(const Target& lhs, const Target& rhs) {
  return (lhs.os_ == rhs.os_) && (lhs.arch_ == rhs.arch_) &&
         (lhs.bitlen_ == rhs.bitlen_);
}
bool operator!=(const Target& lhs, const Target& rhs) {
  return (lhs.os_ != rhs.os_) && (lhs.arch_ != rhs.arch_) &&
         (lhs.bitlen_ != rhs.bitlen_);
}

}  // namespace common
}  // namespace colossalAI
