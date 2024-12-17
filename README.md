# KV-Cache-Compression-via-Token-Reduction
Large Language Models (LLMs) have achieved remarkable success in text generation tasks. However, long-context scenarios, such as Retrieval-Augmented Generation (RAG), introduce significant challenges due to the high computational cost of managing extended input sequences.  

**KV Cache Compression** offers a promising solution by reducing memory usage and computational complexity while maintaining performance. This project explores two primary token compression approaches: 

1. **[LLMLingua](https://github.com/microsoft/LLMLingua)** (state-of-the-art compression method).  
2. **Random Token Pruning** (baseline strategy using random token reduction). 

Additionally, a summarization-based compression strategy was partially implemented but could not be fully evaluated due to computational resource limitations.

Code Structure
• full ds.py: Associated with the baseline that uses the complete, uncompressed context as input to the target LLM.

• llm lingua ds.py: Implements the approach using the LLMLingua tool to compress the context to various target token lengths.

• random prune.py: Handles the baseline where tokens are randomly pruned to achieve a specified token length for the input context.

• summary ds.py: Intended for imple- menting a summarization-based compression strategy, though this method was not fully explored due to computational constraints.
