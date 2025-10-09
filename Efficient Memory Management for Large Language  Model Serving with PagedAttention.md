# Efficient Memory Management for Large Language  Model Serving with PagedAttention

## Abstract
The key-value cache memory for each is huge and grows and shrinks dynamically.
When managed inefficiently,this memory can be significantly wasted by fragementation and redundant duplication,limiting the batch size.
To address this problem,we propose PageAttention ,and attention algorithm inspired by the classical virtual memory and paging techniques in operating systems.
On top of it, we build vLLM, an LLM serving system that achieves **(1) near-zero waste in KV cache memory and (2) flexible 
sharing of KV cache within and across requests to further reduce memory usage**.

##