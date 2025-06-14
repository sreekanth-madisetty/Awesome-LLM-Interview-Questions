## 2. Quantization
**1. What is Quantization in Large Language Models (LLMs)?**

**Answer:**  
Quantization in the context of LLMs is a model compression technique that reduces the precision of a model’s weights and activations. Typically, the parameters in LLMs are stored as high-precision values (such as 32-bit floating-point numbers). Quantization converts these to lower-precision types (such as 8-bit integers or 16-bit floats), thereby reducing the model’s memory footprint and computational requirements.

---

**2. Why to Quantize LLMs?**

**Answer:**  
Quantizing LLMs means making them use less computer power and memory by converting their internal numbers from big, detailed (high-precision) values to smaller, simpler (low-precision) ones. The following are the reasons for quantization in LLMs.

**Smaller Size:** Quantization shrinks the model, so it takes up less storage space and is easier to share or move between devices.

**Faster Performance:** With smaller, simpler numbers, the model can process information and answer questions more quickly—responses come faster.

**Works on More Devices:** Quantized LLMs can run on less powerful hardware, including laptops, phones, and even single GPUs or CPUs, instead of needing special, expensive computers.

**Saves Energy and Money:** Since the model does less work with lower-precision numbers, it uses less electricity and reduces costs, especially when used a lot or by many users

---

**3. What are the types of Quantization?**

**Answer:**  
The main quantization techniques for LLMs include:

**Post-Training Quantization (PTQ):** Quantizes a pre-trained model without retraining, using calibration data to map weights/activations to lower precision. Simple but may lead to accuracy loss.

**Quantization-Aware Training (QAT):** Incorporates quantization constraints during training, simulating low-precision effects to optimize the model. More accurate but computationally expensive.

**Dynamic Quantization:** Quantizes weights statically but activations dynamically during inference, balancing speed and accuracy.

**Mixed-Precision Quantization:** Uses different precisions for different layers or components (e.g., FP16 for attention, INT8 for feed-forward layers) to optimize trade-offs.

---

**4. How does quantization reduce model size?**  

**Answer:**  
Storing parameters in lower precision reduces bits per value. For example:  
- A 7B-parameter model in FP32: 28 GB → 7B parameters × 4 bytes.  
- The same model in INT8: 7 GB → 7B × 1 byte.  

---

**5. What is the trade-off between performance and accuracy in quantization?**  

**Answer:**  
- Lower precision (e.g., INT4) improves speed and reduces size but risks higher accuracy loss.  
- Higher precision (e.g., INT8) balances accuracy and efficiency.  
- Use-case determines the optimal trade-off (e.g., edge devices vs. cloud servers).  

---

**6. How does quantization impact inference latency?**  

**Answer:**  
- **Pros:** Faster computation due to reduced data movement and hardware-optimized low-precision ops.  
- **Cons:** Dequantization overhead in hybrid-precision pipelines.

---
**7. Can you explain the GPTQ algorithm and its relevance to LLM quantization?**

**Answer:**

GPTQ is a PTQ method designed for LLMs. It quantizes weights to low precision (e.g., 4-bit or 3-bit) while minimizing accuracy loss through:

- **Layer-wise Quantization:** Processes one layer at a time to reduce memory usage.
- **Hessian-based Optimization:** Uses second-order information (Hessian matrix) to adjust quantized weights, preserving the model’s output distribution.
- **Error Compensation:** Applies techniques like optimal brain compression to mitigate quantization errors.
- GPTQ is widely used for compressing large models like LLaMA, enabling efficient inference with minimal perplexity degradation.
