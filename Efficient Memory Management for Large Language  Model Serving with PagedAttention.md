# Efficient Memory Management for Large Language  Model Serving with PagedAttention

## Abstract
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
##