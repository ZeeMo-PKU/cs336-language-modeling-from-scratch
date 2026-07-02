# Course Materials

This folder connects this study repository to the official Stanford CS336 lecture materials.

## Official Lecture Materials Submodule

- Local path: `course-materials/stanford-cs336-lectures/`
- Upstream: https://github.com/stanford-cs336/lectures

The upstream repository is included as a Git submodule so the official materials remain clearly attributed and easy to update.

After cloning this repository, initialize the submodule with:

```bash
git submodule update --init --recursive
```

To update the official materials later:

```bash
git submodule update --remote course-materials/stanford-cs336-lectures
```

## Lecture Files

Inside `stanford-cs336-lectures/`, useful entry points include:

```text
lecture_01.py   Overview and tokenization
lecture_02.py   Resource accounting, tensors, FLOPs, memory
lecture_03.pdf  Architecture / Transformer material
lecture_04.pdf  Training language models
lecture_05.pdf  GPUs / hardware material
lecture_06.py   Kernels / systems material
lecture_07.py   Parallelism material
lecture_08.pdf
lecture_09.pdf
lecture_10.py   Inference, KV cache, batching, quantization, serving
lecture_11.pdf
lecture_12.py   Evaluation
lecture_13.py   Data
lecture_14.py   Data / filtering / mixing material
lecture_15.pdf
lecture_16.pdf
lecture_17.py
```

Other useful folders/files:

```text
images/         Figures used in the lectures
assets/         Built website assets
var/traces/     Lecture execution traces
references.py   Reference links used by the lecture scripts
lecture_util.py Helper utilities for lecture rendering
```

## Notes

- Keep personal summaries in `course-notes/`.
- Keep homework implementations in `assignments/`.
- Keep official materials in this submodule instead of editing them directly.

