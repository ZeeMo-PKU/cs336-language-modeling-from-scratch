# CS336: Language Modeling from Scratch

Personal study repository for Stanford CS336: Language Modeling from Scratch.

Official course resources:

- Course website: https://cs336.stanford.edu/
- Stanford Online page: https://online.stanford.edu/courses/cs336-language-modeling-scratch
- Lecture materials: https://github.com/stanford-cs336/lectures
- YouTube playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV

## Repository Layout

```text
course-materials/  Official Stanford CS336 lecture materials submodule
course-notes/      Structured personal lecture notes
notes/             Scratch lecture notes and reading summaries
assignments/       My self-study implementations and homework workspaces
experiments/       Small experiments, profiling notes, and scratch studies
resources/         Curated links and study references
```

## Course Materials

The official Stanford CS336 lecture materials are attached as a Git submodule:

- Local path: `course-materials/stanford-cs336-lectures/`
- Upstream: https://github.com/stanford-cs336/lectures

After cloning this repository, run:

```bash
git submodule update --init --recursive
```

## Study Principles

- Learn by building, not only by watching lectures.
- Track both model quality and systems cost: FLOPs, memory, bandwidth, latency, and throughput.
- Keep official course materials in the upstream submodule; keep my own notes and homework in this repo.
- For assignments, document my own implementation choices, tests, and results.

## Current Focus

- Lecture 1: Overview and tokenization
- Follow-up path: resource accounting, GPUs/kernels, inference, scaling laws, data, and alignment
