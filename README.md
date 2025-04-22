# Awesome-LLM-Interview-Questions
Curated LLM interview questions and answers for data science and AI jobs

## Quantization
### What is Quantization in Large Language Models (LLMs)?
Quantization in the context of LLMs is a model compression technique that reduces the precision of a model’s weights and activations. Typically, the parameters in LLMs are stored as high-precision values (such as 32-bit floating-point numbers). Quantization converts these to lower-precision types (such as 8-bit integers or 16-bit floats), thereby reducing the model’s memory footprint and computational requirements.

### Why to Quantize LLMs?
Quantizing LLMs means making them use less computer power and memory by converting their internal numbers from big, detailed (high-precision) values to smaller, simpler (low-precision) ones. The following are the reasons for quantization in LLMs.

**Smaller Size:** Quantization shrinks the model, so it takes up less storage space and is easier to share or move between devices.

**Faster Performance:** With smaller, simpler numbers, the model can process information and answer questions more quickly—responses come faster.

**Works on More Devices:** Quantized LLMs can run on less powerful hardware, including laptops, phones, and even single GPUs or CPUs, instead of needing special, expensive computers.

**Saves Energy and Money:** Since the model does less work with lower-precision numbers, it uses less electricity and reduces costs, especially when used a lot or by many users
