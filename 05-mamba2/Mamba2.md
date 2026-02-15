## Mamba2 

本项目旨在从零开始 (From Scratch) 使用 OpenAI Triton 复现 Mamba-2 (State Space Duality, SSD) 的核心算子与推理架构。

相比于官方仓库，本项目更加注重代码的可读性和模块化，剥离了复杂的 C++ 绑定，纯粹使用 Python + Triton 实现，非常适合用于深入理解 Mamba-2 的底层计算原理（SSD 算法）以及 Triton 高性能编程。

This project aims to implement the core kernels and inference architecture of Mamba-2 (State Space Duality, SSD) from scratch using OpenAI Triton.

Compared to the official repository, this project prioritizes code readability and modularity. By stripping away complex C++ bindings in favor of a pure Python + Triton implementation, it serves as an ideal resource for gaining a deep understanding of Mamba-2's underlying computational principles (the SSD algorithm) as well as high-performance programming with Triton.