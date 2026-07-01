# Lecture 1: Overview and Tokenization

## 1. What This Lecture Is About

Lecture 1 explains why CS336 teaches language modeling "from scratch" and introduces the first technical component of a language model: tokenization.

The main idea is that modern AI researchers often use higher-level abstractions, such as pretrained models, fine-tuning libraries, or API-based models. These abstractions are productive, but they can hide the underlying mechanisms. CS336 argues that fundamental research still requires understanding the stack below the API.

The course philosophy is:

```text
understanding via building
```

In other words, to understand language models deeply, we should build the major pieces ourselves: tokenizer, Transformer, optimizer, training loop, systems measurements, inference path, and evaluation.

## 2. Why This Course Exists

The lecture describes a shift in how researchers interact with language models:

```text
2016: researchers implemented and trained their own models.
2018: researchers downloaded pretrained models such as BERT and fine-tuned them.
Today: researchers often prompt API models such as GPT, Claude, and Gemini.
```

This movement up the abstraction stack increases productivity. However, the abstractions are leaky. When a model behaves unexpectedly, the cause may be hidden in tokenization, training data, optimization, architecture, inference settings, or system constraints.

Therefore, the course emphasizes full-stack understanding.

## 3. Frontier Models and What Can Transfer

Frontier models are extremely expensive to train and are usually not fully documented. A normal student or research group cannot reproduce a GPT-4-scale model from scratch.

This creates a central teaching problem:

```text
If we can only build small models, what knowledge still transfers to frontier models?
```

The lecture separates transferable knowledge into three categories.

### Mechanics

Mechanics means how things work.

Examples:

- How a Transformer block computes attention and MLP outputs.
- How tokenization maps strings to token IDs.
- How model parallelism works.
- How KV cache supports efficient inference.

Mechanics transfers well because the same basic mechanisms appear in both small and large models.

### Mindset

Mindset means how to think like a large-scale model builder.

Examples:

- Squeeze the most out of available hardware.
- Track FLOPs, memory, bandwidth, latency, and throughput.
- Take scaling seriously.
- Avoid wasting compute at large scale.

This also transfers well. In fact, the larger the model, the more important systems efficiency becomes.

### Intuitions

Intuitions are empirical beliefs about which design choices produce better model quality.

Examples:

- Which data mixture works best.
- Which activation function is better.
- What model depth/width ratio is ideal.
- Which training tricks matter at scale.

These only partially transfer. A decision that works for a small model may not work for a frontier-scale model.

## 4. The Bitter Lesson

The lecture references "The Bitter Lesson": in AI history, methods that can effectively use more computation and data tend to win in the long run.

The wrong interpretation is:

```text
scale is all that matters; algorithms do not matter
```

The right interpretation is:

```text
algorithms that scale are what matter
```

CS336 frames model quality as:

```text
accuracy = efficiency x resources
```

This is not a strict mathematical formula. It is a way to think about model development. Better models come from both more resources and better efficiency.

At large scale, efficiency matters even more because wasted computation becomes extremely expensive.

## 5. Current Language Model Landscape

The lecture sketches the history of language modeling.

Before neural networks, language modeling was mostly statistical:

- Shannon studied the entropy of English.
- N-gram models estimated next-word probabilities from short contexts.

In the neural era, many ingredients accumulated:

- LSTM made recurrent sequence modeling stronger.
- Neural language models introduced learned embeddings.
- Seq2Seq models enabled sequence-to-sequence tasks such as translation.
- Attention allowed models to dynamically focus on relevant context.
- Transformer replaced recurrence with attention-based parallel computation.
- Adam and AdamW made large neural network optimization practical.
- MoE increased model capacity while controlling per-token computation.
- Model parallelism made it possible to train models too large for one GPU.

Modern large language models are built from these ingredients plus large-scale data, compute, and systems engineering.

## 6. Architecture Refinements

The lecture also lists many refinements to the basic Transformer architecture:

- Activation functions: ReLU, SwiGLU.
- Positional encodings: sinusoidal embeddings, RoPE.
- Normalization: LayerNorm, RMSNorm, QK norm, pre-norm, post-norm.
- Attention variants: full attention, sparse/local attention, GQA, MLA.
- Alternative sequence models: linear attention, Mamba, Gated DeltaNet.
- MLP variants: dense MLP and mixture-of-experts MLP.
- Shape choices: hidden dimension, depth, number of heads, number of experts.

The key point is that "Transformer" is not a single frozen design. Modern LLMs are families of related architectures with many engineering and modeling refinements.

## 7. Tokenization

Tokenization is the process of converting raw text into token IDs.

```text
text -> tokens -> token IDs -> embeddings -> Transformer
```

Language models do not directly process human-readable text. They process integer token IDs, which are looked up in an embedding table.

The tokenizer provides two core functions:

```text
encode(text) -> list[int]
decode(list[int]) -> text
```

Tokenization matters because it affects:

- Sequence length.
- Context efficiency.
- Training cost.
- Inference cost.
- Multilingual performance.
- Code and symbol handling.

For example, if one tokenizer represents a sentence using 20 tokens and another uses 35 tokens, the second tokenizer creates more work for the model. This matters because attention cost grows roughly with sequence length.

## 8. BPE Intuition

BPE, or Byte-Pair Encoding, repeatedly merges frequent adjacent units.

Example intuition:

```text
l + o -> lo
lo + w -> low
low + er -> lower
```

Frequent patterns become compact tokens, while rare words can still be represented by smaller pieces.

BPE is useful because it balances:

- Vocabulary size.
- Sequence length.
- Ability to represent rare words.
- Ability to handle multilingual text, code, and symbols.

## 9. What I Should Be Able to Explain

After Lecture 1, I should be able to explain:

- Why CS336 emphasizes building from scratch.
- What a leaky abstraction is in language modeling.
- The difference between mechanics, mindset, and intuitions.
- The correct interpretation of The Bitter Lesson.
- Why efficiency matters more at larger scale.
- Why language models need tokenization.
- How token IDs become embeddings.
- Why BPE reduces sequence length.

## 10. Connections to Assignment 1

Lecture 1 connects directly to Assignment 1.

Important topics to review before coding:

- BPE tokenizer training.
- Encoding and decoding.
- Vocabulary and merges.
- Transformer input embeddings.
- Training loop basics.

The first practical goal is to understand and implement tokenization, because every later part of the language model depends on the token sequence.

