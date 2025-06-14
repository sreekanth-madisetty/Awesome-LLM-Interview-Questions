## Finetuning

**1. What is fine-tuning in the context of LLMs, and why is it performed instead of just using the base pre-trained model with prompting?**

**Answer:**
* **What it is:** Fine-tuning is the process of taking a pre-trained LLM (which has learned general language patterns from vast amounts of text) and further training it on a smaller, specific dataset. This dataset is tailored to a particular task, domain, or desired behavior (like following instructions or adopting a certain style). The process typically involves updating some or all of the model's weights based on this new data.
* **Why it's performed:**
    * **Task Specialization:** Base models are generalists. Fine-tuning allows the model to become an expert on a specific task (e.g., medical text summarization, legal document review, code generation in a specific language) leading to significantly better performance on that task than achievable with prompting alone.
    * **Domain Adaptation:** Models pre-trained on general web text might not understand the nuances, terminology, or style of a specific domain (e.g., finance, scientific research). Fine-tuning on domain-specific data helps the model adapt.
    * **Improved Instruction Following:** Fine-tuning on datasets of instructions and desired responses (Instruction Fine-Tuning) makes models much better at understanding and executing user commands accurately compared to base models.
    * **Style/Persona Adaptation:** Fine-tuning can teach the model to respond in a specific tone, style, or persona required for a particular application (e.g., a formal customer service bot vs. a casual chatbot).
    * **Efficiency at Inference:** While prompting techniques like few-shot learning can guide a model, including examples in every prompt consumes context window space and adds latency. A fine-tuned model has the desired knowledge/behavior baked into its weights, making inference potentially more efficient.
    * **Knowledge Injection (Limited):** While not its primary strength compared to RAG, fine-tuning can help the model implicitly learn and recall specific facts present in the fine-tuning dataset.

---

**2. Can you compare full fine-tuning with Parameter-Efficient Fine-Tuning (PEFT) methods? What are the pros and cons of each?**

**Answer:**
* **Full Fine-Tuning:**
    * **What:** Updates *all* the parameters (weights) of the pre-trained LLM using the specific dataset.
    * **Pros:**
        * Potentially achieves the highest performance/adaptation as all weights can be adjusted.
        * Conceptually simple – it's just continued training.
    * **Cons:**
        * **Computationally Expensive:** Requires significant GPU memory and training time, similar to pre-training but on a smaller scale.
        * **High Storage Cost:** Need to store a complete set of model weights for *each* fine-tuned task, which is prohibitive for large models (e.g., 70B+ parameters).
        * **Risk of Catastrophic Forgetting:** The model might lose some of its general capabilities learned during pre-training while adapting to the new task.

* **Parameter-Efficient Fine-Tuning (PEFT):**
    * **What:** Freezes most of the pre-trained LLM's parameters and adds/modifies only a small number of new or existing parameters. Examples include LoRA, QLoRA, Adapters, Prompt Tuning, Prefix Tuning.
    * **Pros:**
        * **Computationally Efficient:** Requires much less GPU memory and often less training time compared to full fine-tuning. Makes fine-tuning accessible on less powerful hardware.
        * **Low Storage Cost:** Only need to store the small set of modified/added parameters (e.g., LoRA matrices, adapter layers) for each task, which are orders of magnitude smaller than the full model. The base model weights are shared.
        * **Reduced Catastrophic Forgetting:** Since the bulk of the original model is frozen, PEFT methods often preserve general capabilities better.
        * **Modularity:** Easier to manage and deploy multiple task-specific adaptations by swapping out the small PEFT modules.
    * **Cons:**
        * **Potentially Lower Performance Ceiling:** May not achieve the absolute peak performance of full fine-tuning on some tasks, although often comes very close.
        * **Added Complexity:** Introduces new hyperparameters and architectural choices specific to the PEFT method used (e.g., rank in LoRA, bottleneck dimension in Adapters).

---

**3. Explain Low-Rank Adaptation (LoRA). How does it work, and what are its key hyperparameters?**

**Answer:**
* **What it is:** LoRA (Low-Rank Adaptation) is a popular PEFT method. It hypothesizes that the change in weights during model adaptation has a low "intrinsic rank". Therefore, instead of updating the full weight matrix `W`, LoRA approximates the update `ΔW` with a low-rank decomposition.
* **How it Works:**
    1.  **Freeze Original Weights:** The original pre-trained weight matrices (`W`) of the LLM (typically in attention layers and/or feed-forward networks) are kept frozen.
    2.  **Inject Low-Rank Matrices:** For a target weight matrix `W` (e.g., in a self-attention layer), LoRA introduces two smaller matrices, `A` and `B`. The update `ΔW` is represented by their product: `ΔW = B * A`. Here, `A` has dimensions `d x r` and `B` has dimensions `r x k`, where `r` is the rank (and `r << d`, `r << k`).
    3.  **Modify Forward Pass:** The modified forward pass becomes `h = Wx + BAx`. `B` is typically initialized to zeros, so `BAx` is zero at the start, ensuring the initial adapted model is identical to the original model.
    4.  **Train Only A and B:** Only the parameters of the matrices `A` and `B` are trained during fine-tuning. The original weights `W` remain unchanged.
    5.  **Inference:** During inference, the product `BA` can be calculated and added to `W` to get a combined weight matrix (`W' = W + BA`), meaning there's no extra latency compared to the original model (unlike Adapters which add extra layers). Alternatively, the `BAx` calculation can be kept separate.
* **Key Hyperparameters:**
    * **`r` (Rank):** The rank of the decomposition `BA`. This is the most important hyperparameter. A smaller `r` means fewer trainable parameters (more efficient) but potentially less expressive power. A larger `r` increases parameters and expressiveness but may lead to overfitting or diminished efficiency gains. Typical values range from 4, 8, 16 up to 64 or 128.
    * **`alpha` (Scaling Factor):** A scaling factor applied to the LoRA update (`ΔW`). The effective update is often scaled as `(alpha / r) * BA`. This helps tune the magnitude of the adaptation relative to the rank `r`. Often set to be equal to `r` or double `r`.
    * **Target Modules:** Deciding which layers/modules of the LLM to apply LoRA to (e.g., query/key/value projection matrices in attention, feed-forward layers). Applying it to more modules increases trainable parameters but might improve results.

* **QLoRA:** A common optimization where the base model is loaded in 4-bit precision (quantized), further reducing memory, while LoRA adapters (which are still trained in higher precision like 16-bit) are added on top.

---

**4. What is catastrophic forgetting, and how can it be mitigated during fine-tuning?**

**Answer:**
* **What it is:** Catastrophic forgetting is the tendency of a neural network (including LLMs) to abruptly lose knowledge or capabilities learned previously (e.g., during pre-training) when it is trained on a new task or dataset. Fine-tuning solely on a narrow dataset can cause the model to overwrite its general language understanding or performance on tasks not represented in the fine-tuning data.
* **Mitigation Strategies:**
    * **Parameter-Efficient Fine-Tuning (PEFT):** Methods like LoRA, Adapters etc., inherently mitigate forgetting by freezing most of the original weights. Since the core model isn't drastically changed, general knowledge is better preserved.
    * **Replay / Data Mixing:** Mix a small amount of the original pre-training data or data representing general capabilities into the fine-tuning dataset. This constantly reminds the model of its previous knowledge.
    * **Lower Learning Rates:** Using very small learning rates during fine-tuning reduces the magnitude of weight updates, making the process more conservative and less likely to drastically alter existing knowledge.
    * **Multi-Task Fine-tuning:** If fine-tuning for multiple specific tasks, train on a combined dataset covering all tasks simultaneously rather than sequentially.
    * **Regularization Techniques:** Techniques like Elastic Weight Consolidation (EWC), though less common in typical LLM fine-tuning workflows due to complexity, explicitly penalize changes to weights deemed important for previous tasks.
    * **Sequential Fine-tuning with Care:** If sequential fine-tuning is necessary, use PEFT methods and potentially decrease the learning rate for later tasks.

---
**5. What kind of datasets are typically used for fine-tuning LLMs? Give some examples.**

**Answer:**
The type of dataset depends heavily on the goal of fine-tuning:

1.  **Instruction Fine-Tuning Datasets:** Used to improve the model's ability to follow instructions and engage in helpful dialogue. They usually consist of `(instruction, output)` pairs or `(instruction, input, output)` triples.
    * **Examples:** `Alpaca` dataset (generated by Self-Instruct using GPT-3), `Dolly` dataset (human-generated), `OpenAssistant Conversations Dataset (OASST)`, `ShareGPT` data (user-shared ChatGPT conversations).
    * **Format:** Often JSON files where each entry contains an instruction field, an optional input field (for context), and the desired output field.

2.  **Domain-Specific Datasets:** Used to adapt the model to a specific field or knowledge area.
    * **Examples:** A collection of medical research abstracts for summarization, legal contracts for analysis, financial news articles for sentiment analysis, a company's internal documentation for Q&A.
    * **Format:** Can be raw text corpora from the domain, or structured pairs like `(domain_specific_question, answer)`, `(document, summary)`, etc.

3.  **Style/Persona Datasets:** Used to teach the model a specific writing style or conversational persona.
    * **Examples:** A collection of dialogues written in the voice of a specific character, examples of formal technical writing, a dataset of empathetic customer service responses.
    * **Format:** Often conversational or text-based, demonstrating the target style. Could be `(prompt, stylized_response)`.

4.  **Preference Datasets (for RLHF):** While used in RLHF which follows initial fine-tuning, the *data collection* often starts with an SFT model. The data consists of prompts and human-ranked responses, e.g., `(prompt, chosen_response, rejected_response)`. Used to train a reward model, not directly for supervised fine-tuning in the same way.

**Key Consideration:** Data quality is paramount. The fine-tuning dataset should be clean, accurate, diverse, and representative of the target task or domain.

---

**Question 6: How do you choose the right fine-tuning strategy (full vs. specific PEFT method) for a given task and resource constraints?**

**Answer:**
The choice depends on several factors:

1.  **Available Resources (GPU Memory, Time):**
    * **High Resources:** Full fine-tuning might be feasible and potentially yield the best results if you have sufficient high-VRAM GPUs (e.g., multiple A100s/H100s) and time.
    * **Limited Resources:** PEFT methods are the go-to option. QLoRA (using 4-bit quantization) is particularly effective for running on consumer GPUs or smaller setups.

2.  **Number of Tasks:**
    * **Single Task:** Full fine-tuning might be considered if resources allow and peak performance is critical.
    * **Multiple Tasks:** PEFT is highly advantageous. Storing and serving one base model with multiple small PEFT adapters is much more efficient than storing multiple fully fine-tuned models.

3.  **Performance Requirements:**
    * **State-of-the-Art Critical:** Full fine-tuning might have a slight edge in some cases, worth exploring if resources permit.
    * **Good Performance Sufficient:** PEFT methods like LoRA often achieve performance very close (within a few percent) to full fine-tuning, making them suitable for most practical applications.

4.  **Risk of Catastrophic Forgetting:**
    * **High Risk / General Capabilities Important:** PEFT methods are generally safer as they preserve the base model's knowledge better.
    * **Low Risk / Task is Very Narrow:** Full fine-tuning might be acceptable if general capabilities are less critical than specialization.

5.  **Specific Task Nature:**
    * Some evidence suggests certain PEFT methods might be slightly better suited for specific types of adaptation (e.g., prompt/prefix tuning for generation control, LoRA/Adapters for broader knowledge adaptation), but LoRA is often a strong general-purpose choice.

**General Guideline (as of early 2025):** Start with PEFT, particularly (Q)LoRA, due to its efficiency and strong performance. Only consider full fine-tuning if PEFT results are insufficient *and* you have the necessary resources.

---

**7. What are some common challenges encountered when fine-tuning LLMs?**

**Answer:**
Fine-tuning LLMs presents several challenges:

* **Data Quality and Quantity:** Acquiring or creating a high-quality, clean, and sufficiently large dataset specific to the task is often the biggest hurdle. Poor data leads to poor performance. Bias in the data will be learned by the model.
* **Computational Cost:** Even with PEFT, fine-tuning large models requires significant GPU resources (memory and compute), especially for larger ranks or applying adapters to more layers. Full fine-tuning is extremely resource-intensive.
* **Hyperparameter Tuning:** Finding the optimal hyperparameters (learning rate, batch size, number of epochs, PEFT-specific parameters like LoRA rank `r` and `alpha`) can be time-consuming and requires experimentation.
* **Catastrophic Forgetting:** As mentioned, balancing adaptation to the new task with preserving general capabilities is tricky, especially with full fine-tuning.
* **Overfitting:** Fine-tuning datasets are much smaller than pre-training datasets. The model can easily overfit, performing well on the fine-tuning data but poorly on unseen examples of the target task. Regularization, early stopping, and careful validation are needed.
* **Evaluation:** Defining appropriate metrics and evaluation protocols to reliably assess performance on the specific task can be difficult, especially for generative tasks where semantic meaning matters more than exact matches. Human evaluation is often necessary but costly.
* **Alignment and Safety:** Fine-tuning can sometimes inadvertently weaken safety constraints or alignment achieved through earlier RLHF stages if not done carefully. It's important to evaluate for safety regressions after fine-tuning.
* **Choosing the Right Method:** Deciding between full fine-tuning and various PEFT techniques requires understanding the trade-offs.

---

**8. How do you evaluate the performance of a fine-tuned LLM?**

**Answer:**
Evaluating a fine-tuned LLM requires assessing its performance *specifically on the target task or domain* it was fine-tuned for, while also potentially checking for regressions in general capabilities or safety. Methods include:

1.  **Task-Specific Metrics (Automated):**
    * **Classification Tasks:** Accuracy, Precision, Recall, F1-score, AUC on a held-out test set.
    * **Summarization/Translation:** ROUGE, BLEU, METEOR scores comparing generated text to reference text. (Note: These correlate imperfectly with human judgment).
    * **Code Generation:** Pass@k metrics (checking if generated code passes unit tests), code similarity scores.
    * **Question Answering:** Exact Match (EM), F1-score over answers.

2.  **Held-Out Validation/Test Set:** The most fundamental approach. Split your specific fine-tuning dataset (or acquire a separate test set representative of the task) into training, validation (for hyperparameter tuning), and test sets. Monitor loss on the validation set during training and report final metrics on the unseen test set.

3.  **Human Evaluation:** Often essential for generative tasks where automated metrics fall short.
    * **Direct Assessment:** Ask humans to rate the quality of model outputs (e.g., on a Likert scale for fluency, relevance, accuracy, helpfulness).
    * **Pairwise Comparison:** Show humans outputs from the fine-tuned model and a baseline (e.g., the pre-fine-tuning model or a competitor) for the same prompt and ask them to choose the better one (A/B testing). Calculate win rates.
    * **Task Success Rate:** For specific tasks (e.g., generating a usable SQL query, answering a customer question correctly), measure the percentage of time the model successfully completes the task according to human judgment.

4.  **General Capability Benchmarks (Regression Testing):** Optionally, run the fine-tuned model on standard benchmarks (e.g., MMLU, HellaSwag, ARC) to check if fine-tuning caused significant degradation in general reasoning or language understanding (checking the "alignment tax" or catastrophic forgetting).

5.  **Safety and Alignment Evaluation:** Use specialized datasets and prompts (e.g., RealToxicityPrompts, Anthropic's red-teaming prompts) or safety classifiers to check if the fine-tuned model generates harmful, biased, or inappropriate content more often than the baseline.

A combination of automated metrics on a test set and targeted human evaluation is typically the most robust approach.

---

**9. What are the frameworks or libraries used for LLM finetuning?**

The following are some popular frameworks and libraries commonly used for fine-tuning Large Language Models (LLMs):

1.  **Hugging Face Ecosystem:** This is arguably the most dominant ecosystem for LLM fine-tuning.
    * **`transformers`:** The core library providing access to thousands of pre-trained models (like BERT, GPT, Llama, T5, Mistral) and their tokenizers. It includes the powerful `Trainer` API, which simplifies the training loop, handling distributed training, mixed-precision, gradient accumulation, and evaluation.
    * **`peft` (Parameter-Efficient Fine-Tuning):** A Hugging Face library specifically designed to implement various PEFT methods like LoRA (Low-Rank Adaptation), QLoRA (Quantized LoRA), Prefix Tuning, P-Tuning, Prompt Tuning, and Adapters. It integrates seamlessly with `transformers` to allow fine-tuning only a small subset of parameters, saving compute and memory.
    * **`trl` (Transformer Reinforcement Learning):** Originally focused on RLHF (like PPO) and Direct Preference Optimization (DPO), `trl` also includes the `SFTTrainer`, which builds upon the `Trainer` to further simplify Supervised Fine-Tuning (SFT), especially instruction fine-tuning, with features like data packing for efficiency.
    * **`datasets`:** For easily loading, processing, and manipulating datasets, including those commonly used for fine-tuning.
    * **`accelerate`:** Handles distributed training setups (multi-GPU, multi-node, TPUs) and mixed-precision training transparently for PyTorch code, often used under the hood by `Trainer`.
    * **`autotrain-advanced`:** A Hugging Face tool designed to automate the fine-tuning process with minimal code, often configurable via a UI or simple scripts.

2.  **Frameworks Simplifying Fine-Tuning (often built on Hugging Face):**
    * **`Axolotl`:** A popular tool that provides a configuration-driven (YAML) approach to fine-tuning various LLMs using PEFT methods. It wraps Hugging Face libraries, aiming for ease of use while retaining flexibility, and supports multi-GPU training well. Often recommended for beginners.
    * **`Unsloth`:** Focuses heavily on optimizing fine-tuning speed and reducing memory usage (claiming 2-5x speedups and 80% less memory) without quantization, using custom Triton kernels for layers like attention. Excellent for resource-constrained environments (e.g., single consumer GPUs) but may lack multi-GPU support.
    * **`Torchtune`:** A PyTorch-native library specifically for fine-tuning LLMs. It offers less abstraction than `Trainer` or `Axolotl`, appealing to users who prefer working directly with PyTorch code. Focuses on being lean, extensible, and memory-efficient.
    * **`LLaMA-Factory`:** An easy-to-use framework, often highlighted for its user-friendly GUI, supporting fine-tuning of various models (Llama, Falcon, Qwen, etc.) and integrating optimizations like Unsloth.
    * **`lit-gpt`:** A hackable and straightforward implementation of various LLMs based on nanoGPT, supporting pre-training, fine-tuning (LoRA, Adapters), quantization, and Flash Attention.
  
