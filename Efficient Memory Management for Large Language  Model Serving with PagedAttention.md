# Efficient Memory Management for Large Language  Model Serving with PagedAttention

## 1 Abstract
The key-value cache memory for each is huge and grows and shrinks dynamically.
When managed inefficiently,this memory can be significantly wasted by fragementation and redundant duplication,limiting the batch size.
To address this problem,we propose PageAttention ,and attention algorithm inspired by the classical virtual memory and paging techniques in operating systems.
On top of it, we build vLLM, an LLM serving system that achieves 
**(1) near-zero waste in KV cache memory and (2) flexible  sharing of KV cache within and across requests to further reduce memory usage**.

the memory distribution for a 13B-parameter LLM on an NVIDIA A100 GPU with 40GB RAM. 
Approximately 65% of the memory is allocated for the model weights, which remain static during serving. 
Close to 30% of the memory is used to store the dynamic states of the requests.
For Transformers, these states consist of the key and value tensors associated with the attention mechanism, commonly referred to as KV cache
which represent the context from earlier tokens to generate new output tokens in sequence.
The remaining small percentage of memory is used for other data.
including activations --the ephemeral tensors created when evaluating the LLM.


KV caches feature:it dynamically grows and shrinks over time as the
model generates new tokens, and its lifetime and length are
not known a priori


**internal and external memory fragmentation:**
**internal fragmentation:**
To store the KV cache of a request in contiguous space,the existing systems pre-allocate a contiguous chunk of memory with the request’s maximum length.
This can result in severe internal fragmentation, 
since the request’s actual length can be much shorter than its maximum length.
**external memory fragmentation:**
 external memory fragmentation can also be significant, since the preallocated size can be different for each request.

the existing systems cannot exploit the opportunities for memory sharing.
decoding algorithms, such as parallel sampling and beam
search, that generate multiple outputs per request.

在多输出（beam search / parallel sampling）场景下，
多个生成序列在早期阶段共享相同上下文，因此其 KV Cache 可以共享。

传统系统无法利用这种共享机会（每个序列独立分配 KV Cache），
导致显存浪费；
而像 PagedAttention 这样的系统可以在页粒度上共享 KV Cache，
显著提升显存效率与吞吐率
## 2 Background

### 2.1 Transformer-Based Large Language Models
### 2.2 LLM Service & Autoregressive Generation

**The prompt phase**
the computation of the prompt phase can be parallelized using matrixmatrix multiplication operations.
**The autoregressive generation phase**
The computation at different iterations cannot be parallelized due to the data dependency and often uses matrix-vector multiplication,which is less efficient.
### 2.3 Batching Techniques for LLMs

A straightforward batching technique would pad the inputs and outputs of the requests to equalize their lengths, wasting GPU computation and memory.
solution to this problem:iteration-level scheduling.
After each iteration, completed requests are removed from the batch, and new ones are added. 
Therefore, a new request can be processed after waiting for a single iteration, not waiting for the entire batch to complete.

## 3 Memory Challenges in LLM Serving
### Large KV cache
### compelx decoding algorithms

The extent of KV cache sharing depends on the specific decoding algorithm employed.
### Scheduling for unknow input & output lengths

### 3.1 Memory Management in Existing Systems

## 4 Method

Each block table entry
records the corresponding physical blocks of a logical block
and the number of filled positions

shared memory:vLLM implements a copy-onwrite mechanism at the block granularity for the physical
blocks that need modification by multiple sequences, similar
to the copy-on-write technique in OS virtual memory 
**Parallel sampling**
In summary, vLLM enables the sharing of most of the
space used to store the prompts’ KV cache across multiple
output samples, with the exception of the final logical block,
which is managed by a copy-on-write mechanism. By sharing
physical blocks across multiple samples, memory usage can
be greatly reduced, especially for long input prompts.
**Beam search**:
