# KV-Cache-Compression-via-Token-Reduction
KV Cache Compression via Token Reduction in Long-Context Scenarios 

• full ds.py: Associated with the baseline that uses the complete, uncompressed context as input to the target LLM.

• llm lingua ds.py: Implements the approach using the LLMLingua tool to compress the context to various target token lengths.

• random prune.py: Handles the baseline where tokens are randomly pruned to achieve a specified token length for the input context.

• summary ds.py: Intended for imple- menting a summarization-based compression strategy, though this method was not fully ex- plored due to computational constraints.
