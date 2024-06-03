# Change Log

All notable changes to this project will be documented in this file.

🚩 **We have moved the change log to the GitHub [release page](https://github.com/hpcaitech/ColossalAI/releases)**

## v0.0.2 | 2022-02

### Added

- Unified distributed layers
- MoE support
- DevOps tools such as github action, code review automation, etc.
- New project official website

### Changes

- refactored the APIs for usability, flexibility and modularity
- adapted PyTorch AMP for tensor parallel
- refactored utilities for tensor parallel and pipeline parallel
- Separated benchmarks and examples as independent repositories
- Updated pipeline parallelism to support non-interleaved and interleaved versions
- refactored installation scripts for convenience

### Fixed

- zero level 3 runtime error
- incorrect calculation in gradient clipping


## v0.0.1 beta | 2021-10

The first beta version of Colossal-AI. Thanks to all contributors for the effort to implement the system.

### Added

- Initial architecture of the system
- Features such as tensor parallelism, gradient clipping, gradient accumulation
