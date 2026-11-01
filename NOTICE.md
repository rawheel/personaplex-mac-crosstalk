# Notices

## Code

- `goyaah/model/` is adapted from Kyutai's
  [moshi_mlx](https://github.com/kyutai-labs/moshi) (MIT), which includes code from Apple's
  [mlx-examples](https://github.com/ml-explore/mlx-examples) (MIT). Changes are described at
  the top of each file.
- Those files keep their original notices at the top: Copyright (c) Kyutai, and Copyright ©
  2023-2024 Apple Inc. Both are under the MIT license, the same terms as [LICENSE](LICENSE).
- `goyaah/engine.py` ports the streaming logic of NVIDIA's
  [PersonaPlex](https://github.com/NVIDIA/personaplex) `LMGen` (MIT), and `goyaah/convert.py`
  follows the parameter mapping of Kyutai's `scripts/import_mlx.py` (MIT).

## Model weights

The model weights are **not** part of this repository. `goyaah-convert` downloads
[nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) from Hugging Face,
which requires accepting its license.

PersonaPlex is licensed by NVIDIA Corporation under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
It is built on Kyutai's Moshi weights (CC-BY 4.0). If you redistribute converted weights,
include the NVIDIA Open Model License and this notice with them.
