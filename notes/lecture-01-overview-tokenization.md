# Lecture 1: Overview and Tokenization

## Main Question

Why should we study language modeling from scratch, and why does tokenization come first?

## Key Ideas

- Modern researchers often work at higher abstraction levels: APIs, fine-tuning, and prompt engineering.
- These abstractions are productive but leaky.
- CS336 emphasizes understanding via building.
- Mechanics and systems mindset transfer better to frontier models than small-scale intuitions.
- Tokenization maps raw text into token IDs, which are the actual inputs to a language model.

## Terms

- Mechanics: how components work.
- Mindset: how to reason about scaling, systems, and hardware efficiency.
- Intuitions: empirical guesses about data and modeling choices.
- Tokenizer: `encode(text) -> token_ids`, `decode(token_ids) -> text`.
- BPE: Byte-Pair Encoding, a common subword tokenization strategy.

## Study Questions

1. Why is a language model not simply a text-processing program?
2. What makes an abstraction "leaky" in language modeling?
3. Which parts of small-model training transfer to frontier models?
4. Why does tokenization affect compute and context efficiency?

