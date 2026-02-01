# Hardware and Compute Requirements

NAAMSE is designed to run efficiently on consumer-grade hardware, requiring only basic CPU and GPU capabilities for the core framework operations. The system does not demand high-end computing resources, as most computationally intensive tasks (such as LLM inference for prompt mutation and behavioral scoring) are handled via external API calls rather than local processing.

## Minimum Requirements

- **CPU**: Quad-core processor (e.g., Intel i5 or equivalent)
- **RAM**: 8 GB
- **Storage**: 10 GB free space (for databases and dependencies)
- **GPU**: Recommended for faster embedding computations; 4GB VRAM suffices (e.g., NVIDIA GTX 1050 or equivalent). Not required but improves performance.
- **OS**: Linux, macOS, or Windows with Python 3.10+

## LLM Inference and API Costs

While the framework itself is relatively lightweight, NAAMSE relies on Large Language Model (LLM) APIs for:
- Prompt mutation and generation
- Behavioral response scoring and jailbreak detection
- Embedding computations for similarity search

**API Requirements**:
- Access to an LLM provider (e.g., Google Gemini, OpenAI)
- Valid API key with sufficient quota
- Internet connection for API calls

**Cost Considerations**:
- Evaluation runs incur API costs proportional to the number of iterations, mutations, and agent invocations.
- For example, a typical evaluation with 7 iterations and 4 mutations per iteration may generate hundreds of API calls.
- Costs vary by provider (e.g., $0.001-0.01 per 1K tokens for Gemini).
- Budget accordingly based on evaluation frequency and scale.

This design offloads heavy inference to cloud providers, ensuring NAAMSE remains accessible on standard hardware while leveraging state-of-the-art AI capabilities.