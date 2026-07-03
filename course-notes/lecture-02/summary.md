# Lecture 2: PyTorch, Resource Accounting, and Gradients

## 1. What This Lecture Is About

Lecture 2 teaches how to think quantitatively about language models.

The key idea is:

```text
Do not only ask what a model computes.
Ask how many FLOPs it costs, how much memory it uses, and what hardware bottleneck it hits.
```

This lecture builds the foundation for later systems topics:

- GPU efficiency
- training cost estimation
- inference bottlenecks
- memory-bound versus compute-bound operations
- gradient computation cost
- large-batch training techniques

## 2. Tensor Shapes

Most language model code is tensor shape manipulation.

Common dimensions:

```text
batch       number of examples processed together
seq         sequence length
hidden      hidden dimension / embedding dimension
head        attention head index
head_dim    dimension inside each attention head
vocab       vocabulary size
in/out      input and output feature dimensions
```

Examples:

```text
token_ids:          batch seq
embeddings:         batch seq hidden
attention scores:   batch head seq seq
logits:             batch seq vocab
```

Understanding tensor shapes is necessary before understanding Transformer implementation.

## 3. einops and einsum

`einops` lets us write tensor operations using named dimensions.

Useful functions:

```text
rearrange   reshape / transpose / split / merge dimensions
reduce      sum / mean / max / min over named dimensions
repeat      introduce or copy dimensions
einsum      generalized matrix multiplication with named dimensions
```

Example:

```python
einsum(x, w, "batch in, in out -> batch out")
```

This means:

```text
x:      batch in
w:      in out
output: batch out
```

The `in` dimension appears in the inputs but not in the output, so it is summed over.

This is equivalent to:

```python
x @ w
```

In attention, a typical pattern is:

```python
einsum(q, k, "batch head seq_q dim, batch head seq_k dim -> batch head seq_q seq_k")
```

This computes dot products between query tokens and key tokens.

## 4. Bytes and Data Types

Memory cost depends on dtype.

Common sizes:

```text
float32 / FP32      4 bytes
float16 / FP16      2 bytes
bfloat16 / BF16     2 bytes
int8                1 byte
int4                0.5 byte
```

Example:

```text
70B parameters in FP16 = 70B * 2 bytes = 140 GB
```

This only counts weights, not gradients, optimizer states, activations, KV cache, or temporary buffers.

## 5. FLOPs

FLOPs means floating-point operations.

Important distinction:

```text
FLOPs   total amount of computation
FLOP/s  computation speed of hardware
```

For matrix multiplication:

```text
(m, k) @ (k, n) -> (m, n)
FLOPs ≈ 2 * m * k * n
```

The factor 2 comes from multiplication plus addition.

Example:

```text
h2 = h1 @ w2
h1: (B, D)
w2: (D, D)
h2: (B, D)
```

Forward FLOPs:

```text
2 * B * D * D
```

For `B = 1024` and `D = 256`:

```text
2 * 1024 * 256 * 256 = 134,217,728 FLOPs
```

## 6. MFU

MFU means Model FLOPs Utilization.

It compares actual measured speed with hardware peak speed:

```text
MFU = actual FLOP/s / promised FLOP/s
```

If a GPU theoretically supports 1000 TFLOP/s and the workload achieves 500 TFLOP/s:

```text
MFU = 0.5
```

An MFU around or above 0.5 is often considered good for large training workloads.

MFU is usually below 1 because of memory traffic, communication, kernel overhead, non-matmul operations, imperfect tensor shapes, and framework overhead.

## 7. Arithmetic Intensity

Arithmetic intensity measures how much computation is done per byte moved:

```text
arithmetic intensity = FLOPs / bytes moved
```

Hardware has a corresponding threshold:

```text
accelerator intensity = peak FLOP/s / memory bandwidth
```

Decision rule:

```text
arithmetic intensity < accelerator intensity -> memory-bound
arithmetic intensity > accelerator intensity -> compute-bound
```

For H100 in the lecture:

```text
peak FLOP/s ≈ 1979e12 / 2
memory bandwidth ≈ 3.35e12 bytes/s
accelerator intensity ≈ 295 FLOPs/byte
```

So a workload needs roughly 295 FLOPs per byte moved to keep the H100 compute units fully busy.

## 8. Memory-Bound Examples

### ReLU

ReLU:

```text
ReLU(x) = max(0, x)
```

For BF16:

```text
read x:  2n bytes
write y: 2n bytes
total:   4n bytes
```

FLOPs:

```text
n comparisons
```

Arithmetic intensity:

```text
n / 4n = 0.25 FLOPs/byte
```

This is far below H100 accelerator intensity, so isolated ReLU is memory-bound.

### GeLU

GeLU does more computation per element than ReLU, so its arithmetic intensity is higher.

But it is still usually memory-bound when run in isolation.

## 9. Roofline Model

The roofline model visualizes the relationship between arithmetic intensity and achievable performance.

Core formula:

```text
achievable FLOP/s = min(peak FLOP/s, bandwidth * arithmetic intensity)
```

The plot has:

```text
x-axis: arithmetic intensity
y-axis: realized FLOP/s
```

The kink is:

```text
accelerator intensity = peak FLOP/s / bandwidth
```

Left of the kink:

```text
memory-bound
```

Right of the kink:

```text
compute-bound
```

Idealized MFU can be related to arithmetic intensity:

```text
MFU = min(1, arithmetic_intensity / accelerator_intensity)
```

This explains why elementwise operations can have very low MFU while large matrix multiplications can approach the compute ceiling.

## 10. Why This Matters for LLMs

LLMs include many operation types:

```text
large matmuls
matrix-vector products
elementwise activations
normalization
softmax
attention
KV cache reads/writes
```

Training and prefill often involve large matrix-matrix multiplications:

```text
high arithmetic intensity
more likely compute-bound
```

Decode often resembles matrix-vector multiplication:

```text
low arithmetic intensity
more likely memory-bound
```

This is why inference optimization focuses heavily on batching, KV cache management, quantization, operator fusion, and memory bandwidth.

## 11. Gradients and Backward Pass

Training requires gradients.

Basic flow:

```text
forward:   compute prediction and loss
backward:  compute gradients
update:    optimizer changes parameters
```

Example:

```python
w = torch.tensor([1., 1., 1.], requires_grad=True)
loss.backward()
```

`requires_grad=True` tells PyTorch:

```text
track computations involving this tensor and compute loss gradients for it
```

After `loss.backward()`, PyTorch stores gradients in:

```text
w.grad
```

For model parameters, gradients tell the optimizer how to change the parameters to reduce loss.

## 12. Backward FLOPs

For a linear layer:

```text
h2 = h1 @ w2
```

Shapes:

```text
h1: (B, D)
w2: (D, D)
h2: (B, D)
```

Forward requires one matmul:

```text
h2 = h1 @ w2
FLOPs = 2 * B * D * D
```

Backward needs two matmuls:

```text
h1.grad = h2.grad @ w2.T
w2.grad = h1.T @ h2.grad
```

Each costs:

```text
2 * B * D * D
```

So:

```text
Backward FLOPs = 4 * B * D * D
```

Thus:

```text
backward is about 2x forward
```

## 13. Training FLOPs Approximation

For MLP-like networks:

```text
Forward pass:  2 * (# data points) * (# parameters)
Backward pass: 4 * (# data points) * (# parameters)
Total:         6 * (# data points) * (# parameters)
```

For language models, a useful rough approximation is:

```text
training FLOPs ≈ 6 * (# tokens) * (# parameters)
```

Example:

```text
70B parameters trained on 15T tokens
FLOPs ≈ 6 * 70B * 15T
```

This is a good approximation for Transformers with short context lengths. For long context lengths, attention's sequence-length-squared cost becomes more important.

## 14. Gradient Accumulation

Large batch sizes can improve training stability, but activation memory scales with batch size.

Gradient accumulation solves this by splitting one large batch into smaller micro-batches.

Instead of:

```text
64 samples -> forward/backward -> update once
```

Use:

```text
16 samples -> backward, accumulate gradient
16 samples -> backward, accumulate gradient
16 samples -> backward, accumulate gradient
16 samples -> backward, accumulate gradient
update once
```

This simulates batch size 64 while only storing activations for 16 samples at a time.

Important distinction:

```text
parameter gradient memory depends mostly on parameter count
activation memory scales with micro-batch size
```

Gradients are not stored as four separate copies. They are accumulated into the same `.grad` buffer:

```text
param.grad += micro_batch_grad
```

Usually the loss is divided by accumulation steps:

```python
loss = loss / accumulation_steps
loss.backward()
```

This makes the accumulated gradient match the average large-batch gradient.

## 15. Training Loop

A minimal training loop:

```python
for batch in dataloader:
    optimizer.zero_grad()

    pred = model(batch)
    loss = loss_fn(pred, target)

    loss.backward()
    optimizer.step()
```

The loop in the lecture follows:

```text
get data
forward
compute loss
backward
optimizer.step()
optimizer.zero_grad()
```

`optimizer.step()` updates parameters using gradients.

`optimizer.zero_grad(set_to_none=True)` clears old gradients because PyTorch accumulates gradients by default.

## 16. Main Takeaways

After Lecture 2, I should be able to:

- Read tensor shapes and map them to model dimensions.
- Understand simple `einops` expressions.
- Estimate matrix multiplication FLOPs.
- Estimate tensor memory from dtype and shape.
- Distinguish FLOPs from FLOP/s.
- Compute arithmetic intensity.
- Use the roofline model to identify memory-bound versus compute-bound workloads.
- Explain why backward is more expensive than forward.
- Use the `6 * tokens * parameters` training FLOPs approximation.
- Explain gradient accumulation and why it saves activation memory.
- Understand the standard training loop.

