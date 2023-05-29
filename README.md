


# Open-LLM-datasets
Repository for organizing datasets used in Open LLM. To download or access information about the most commonly used datasets, please refer to Hugging Face.
https://huggingface.co/datasets

<br>

# References
[1]https://github.com/KennethanCeyer/awesome-llm <br>
[2]https://github.com/Hannibal046/Awesome-LLM <br>
[3]https://github.com/Zjh-819/LLMDataHub <br>
[4]https://huggingface.co/datasets <br>


<br>

# Contents
- [Open-LLM-datasets](#open-llm-datasets)
- [References](#references)
- [Contents](#contents)
- [Papers](#papers)
  - [Pre-trained LLM](#pre-trained-llm)
  - [Instruction finetuned LLM](#instruction-finetuned-llm)
  - [Aligned LLM](#aligned-llm)
- [Open LLM](#open-llm)
  - [LLM Training Frameworks](#llm-training-frameworks)
  - [Tools for deploying LLM](#tools-for-deploying-llm)
  - [Tutorials about LLM](#tutorials-about-llm)
  - [Courses about LLM](#courses-about-llm)
  - [Opinions about LLM](#opinions-about-llm)
  - [Other Awesome Lists](#other-awesome-lists)
  - [Other Useful Resources](#other-useful-resources)
  - [General Open Access Datasets for Alignment](#general-open-access-datasets-for-alignment)
  - [Open Datasets for Pretraining](#open-datasets-for-pretraining)
  - [Domain-specific Datasets and Private dataset](#domain-specific-datasets-and-private-dataset)
  - [Potential Overlap](#potential-overlap)
  - [Contribute](#contribute)


# Papers

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf)
- [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)
- [Language models are few-shot learners](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf)
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374.pdf)
- [On the Opportunities and Risks of Foundation Models](https://arxiv.org/pdf/2108.07258.pdf)
- [Finetuned Language Models are Zero-Shot Learners](https://openreview.net/forum?id=gEZrGCozdqR)
- [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/pdf/2112.06905.pdf)
- [WebGPT: Improving the Factual Accuracy of Language Models through Web Browsing](https://openai.com/blog/webgpt/)
- [Improving language models by retrieving from trillions of tokens](https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens)
- [Scaling Language Models: Methods, Analysis &amp; Insights from Training Gopher](https://arxiv.org/pdf/2112.11446.pdf)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf)
- [LaMDA: Language Models for Dialog Applications](https://arxiv.org/pdf/2201.08239.pdf)
- [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858)
- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/pdf/2201.11990.pdf)
- [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311.pdf)
- [An empirical analysis of compute-optimal large language model training](https://www.deepmind.com/publications/an-empirical-analysis-of-compute-optimal-large-language-model-training)
- [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/2205.01068.pdf)
- [Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131v1)
- [Emergent Abilities of Large Language Models](https://openreview.net/pdf?id=yzkSU5zdwD)
- [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://github.com/google/BIG-bench)
- [Language Models are General-Purpose Interfaces](https://arxiv.org/pdf/2206.06336.pdf)
- [Improving alignment of dialogue agents via targeted human judgements](https://arxiv.org/pdf/2209.14375.pdf)
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf)
- [GLM-130B: An Open Bilingual Pre-trained Model](https://arxiv.org/pdf/2210.02414.pdf)
- [Holistic Evaluation of Language Models](https://arxiv.org/pdf/2211.09110.pdf)
- [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/pdf/2211.05100.pdf)
- [Galactica: A Large Language Model for Science](https://arxiv.org/pdf/2211.09085.pdf)
- [OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization](https://arxiv.org/pdf/2212.12017)
- [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/pdf/2301.13688.pdf)
- [LLaMA: Open and Efficient Foundation Language Models](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/)
- [Language Is Not All You Need: Aligning Perception with Language Models](https://arxiv.org/abs/2302.14045)
- [PaLM-E: An Embodied Multimodal Language Model](https://palm-e.github.io)
- [GPT-4 Technical Report](https://openai.com/research/gpt-4)
- [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)
- [Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision](https://arxiv.org/abs/2305.03047)
- [PaLM 2 Technical Report](https://ai.google/static/documents/palm2techreport.pdf)
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)


## Pre-trained LLM
- Switch Transformer: [Paper](https://arxiv.org/pdf/2101.03961.pdf)
- GLaM: [Paper](https://arxiv.org/pdf/2112.06905.pdf)
- PaLM: [Paper](https://arxiv.org/pdf/2204.02311.pdf)
- MT-NLG: [Paper](https://arxiv.org/pdf/2201.11990.pdf)
- J1-Jumbo: [api](https://docs.ai21.com/docs/complete-api), [Paper](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)
- OPT: [api](https://opt.alpa.ai), [ckpt](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT), [Paper](https://arxiv.org/pdf/2205.01068.pdf), [OPT-175B License Agreement](https://github.com/facebookresearch/metaseq/blob/edefd4a00c24197486a3989abe28ca4eb3881e59/projects/OPT/MODEL_LICENSE.md)
- BLOOM: [api](https://huggingface.co/bigscience/bloom), [ckpt](https://huggingface.co/bigscience/bloom), [Paper](https://arxiv.org/pdf/2211.05100.pdf), [BigScience RAIL License v1.0](https://huggingface.co/spaces/bigscience/license)
- GPT 3.0: [api](https://openai.com/api/), [Paper](https://arxiv.org/pdf/2005.14165.pdf)
- LaMDA: [Paper](https://arxiv.org/pdf/2201.08239.pdf)
- GLM: [ckpt](https://github.com/THUDM/GLM-130B), [Paper](https://arxiv.org/pdf/2210.02414.pdf), [The GLM-130B License](https://github.com/THUDM/GLM-130B/blob/799837802264eb9577eb9ae12cd4bad0f355d7d6/MODEL_LICENSE)
- YaLM: [ckpt](https://github.com/yandex/YaLM-100B), [Blog](https://medium.com/yandex/yandex-publishes-yalm-100b-its-the-largest-gpt-like-neural-network-in-open-source-d1df53d0e9a6), [Apache 2.0 License](https://github.com/yandex/YaLM-100B/blob/14fa94df2ebbbd1864b81f13978f2bf4af270fcb/LICENSE)
- LLaMA: [ckpt](https://github.com/facebookresearch/llama), [Paper](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/), [Non-commercial bespoke license](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)
- GPT-NeoX: [ckpt](https://github.com/EleutherAI/gpt-neox), [Paper](https://arxiv.org/pdf/2204.06745.pdf), [Apache 2.0 License](https://github.com/EleutherAI/gpt-neox/blob/main/LICENSE)
- UL2: [ckpt](https://huggingface.co/google/ul2), [Paper](https://arxiv.org/pdf/2205.05131v1.pdf), [Apache 2.0 License](https://huggingface.co/google/ul2)
- T5: [ckpt](https://huggingface.co/t5-11b), [Paper](https://jmlr.org/papers/v21/20-074.html), [Apache 2.0 License](https://huggingface.co/t5-11b)
- CPM-Bee: [api](https://live.openbmb.org/models/bee), [Paper](https://arxiv.org/pdf/2012.00413.pdf)
- rwkv-4: [ckpt](https://huggingface.co/BlinkDL/rwkv-4-pile-7b), [Github](https://github.com/BlinkDL/RWKV-LM), [Apache 2.0 License](https://huggingface.co/BlinkDL/rwkv-4-pile-7b)
- GPT-J: [ckpt](https://huggingface.co/EleutherAI/gpt-j-6B), [Github](https://github.com/kingoflolz/mesh-transformer-jax), [Apache 2.0 License](https://huggingface.co/EleutherAI/gpt-j-6b)
- GPT-Neo: [ckpt](https://github.com/EleutherAI/gpt-neo), [Github](https://github.com/EleutherAI/gpt-neo), [MIT License](https://github.com/EleutherAI/gpt-neo/blob/23485e3c7940560b3b4cb12e0016012f14d03fc7/LICENSE)



## Instruction finetuned LLM
- Flan-PaLM: [Link](https://arxiv.org/pdf/2210.11416.pdf)
- BLOOMZ: [Link](https://huggingface.co/bigscience/bloomz)
- InstructGPT: [Link](https://platform.openai.com/overview)
- Galactica: [Link](https://huggingface.co/facebook/galactica-120b)
- OpenChatKit: [Link](https://github.com/togethercomputer/OpenChatKit)
- Flan-UL2: [Link](https://github.com/google-research/google-research/tree/master/ul2)
- Flan-T5: [Link](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)
- T0: [Link](https://huggingface.co/bigscience/T0)
- Alpaca: [Link](https://crfm.stanford.edu/alpaca/)


## Aligned LLM
- GPT 4: [Blog](https://openai.com/research/gpt-4)
- ChatGPT: [Demo](https://openai.com/blog/chatgpt/) | [API](https://share.hsforms.com/1u4goaXwDRKC9-x9IvKno0A4sk30)
- Sparrow: [Paper](https://arxiv.org/pdf/2209.14375.pdf)
- Claude: [Demo](https://poe.com/claude) | [API](https://www.anthropic.com/earlyaccess)


<br><br>

# Open LLM

- [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) - A foundational, 65-billion-parameter large language model.
- [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) - A model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations.
- [Flan-Alpaca](https://github.com/declare-lab/flan-alpaca) - Instruction Tuning from Humans and Machines.
- [Baize](https://github.com/project-baize/baize-chatbot) - Baize is an open-source chat model trained with LoRA.
- [Cabrita](https://github.com/22-hours/cabrita) - A Portuguese finetuned instruction LLaMA.
- [Vicuna](https://github.com/lm-sys/FastChat) - An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality.
- [Llama-X](https://github.com/AetherCortex/Llama-X) - Open Academic Research on Improving LLaMA to SOTA LLM.
- [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna) - A Chinese Instruction-following LLaMA-based Model.
- [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) - 4 bits quantization of LLaMA using GPTQ.
- [GPT4All](https://github.com/nomic-ai/gpt4all) - Demo, data, and code to train open-source assistant-style large language model based on GPT-J and LLaMa.
- [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/) - A Dialogue Model for Academic Research.
- [BELLE](https://github.com/LianjiaTech/BELLE) - Be Everyone's Large Language model Engine.
- [StackLLaMA](https://huggingface.co/blog/stackllama) - A hands-on guide to train LLaMA with RLHF.
- [RedPajama](https://github.com/togethercomputer/RedPajama-Data) - An Open Source Recipe to Reproduce LLaMA training dataset.
- [Chimera](https://github.com/FreedomIntelligence/LLMZoo) - Latin Phoenix.
- [CaMA](https://github.com/zjunlp/CaMA) - a Chinese-English Bilingual LLaMA Model.
- [BLOOM](https://huggingface.co/bigscience/bloom) - BigScience Large Open-science Open-access Multilingual Language Model.
- [BLOOMZ&mT0](https://huggingface.co/bigscience/bloomz) - a family of models capable of following human instructions in dozens of languages zero-shot.
- [Phoenix](https://github.com/FreedomIntelligence/LLMZoo)
- [T5](https://arxiv.org/abs/1910.10683) - Text-to-Text Transfer Transformer.
- [T0](https://arxiv.org/abs/2110.08207) - Multitask Prompted Training Enables Zero-Shot Task Generalization.
- [OPT](https://arxiv.org/abs/2205.01068) - Open Pre-trained Transformer Language Models.
- [UL2](https://arxiv.org/abs/2205.05131v1) - a unified framework for pretraining models that are universally effective across datasets and setups.
- [GLM](https://github.com/THUDM/GLM)- GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on various natural language understanding and generation tasks.
- [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) - ChatGLM-6B is an open-source, supporting Chinese and English dialogue language model based on General Language Model (GLM) architecture.
- [RWKV](https://github.com/BlinkDL/RWKV-LM) - Parallelizable RNN with Transformer-level LLM Performance.
- [ChatRWKV](https://github.com/BlinkDL/ChatRWKV) - ChatRWKV is like ChatGPT but powered by my RWKV (100% RNN) language model.
- [StableLM](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models) - Stability AI Language Models.
- [YaLM](https://medium.com/yandex/yandex-publishes-yalm-100b-its-the-largest-gpt-like-neural-network-in-open-source-d1df53d0e9a6) - a GPT-like neural network for generating and processing text.
- [GPT-Neo](https://github.com/EleutherAI/gpt-neo) - An implementation of model & data parallel GPT3-like models.
- [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax/#gpt-j-6b) - A 6 billion parameter, autoregressive text generation model trained on The Pile.
- [Dolly](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html) - a cheap-to-build LLM that exhibits a surprising degree of the instruction following capabilities exhibited by ChatGPT.
- [Pythia](https://github.com/EleutherAI/pythia) - Interpreting Autoregressive Transformers Across Time and Scale.
- [Dolly 2.0](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) - the first open source, instruction-following LLM, fine-tuned on a human-generated instruction dataset licensed for research and commercial use.
- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) - an open-source reproduction of DeepMind's Flamingo model.
- [Cerebras-GPT](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/) - A Family of Open, Compute-efficient, Large Language Models.
- [GALACTICA](https://github.com/paperswithcode/galai/blob/main/docs/model_card.md) - The GALACTICA models are trained on a large-scale scientific corpus.
- [GALPACA](https://huggingface.co/GeorgiaTechResearchInstitute/galpaca-30b) - GALACTICA 30B fine-tuned on the Alpaca dataset.
- [Palmyra](https://huggingface.co/Writer/palmyra-base) - Palmyra Base was primarily pre-trained with English text.
- [Camel](https://huggingface.co/Writer/camel-5b-hf) - a state-of-the-art instruction-following large language model.
- [h2oGPT](https://github.com/h2oai/h2ogpt)
- [PanGu-α](https://openi.org.cn/pangu/) - PanGu-α is a 200B parameter autoregressive pretrained Chinese language model.
- [MOSS](https://github.com/OpenLMLab/MOSS) - MOSS is an open-source dialogue language model that supports Chinese and English.
- [Open-Assistant](https://github.com/LAION-AI/Open-Assistant) - a project meant to give everyone access to a great chat-based large language model.
- [HuggingChat](https://huggingface.co/chat/) - Powered by Open Assistant's latest model – the best open-source chat model right now and @huggingface Inference API.
- [StarCoder](https://huggingface.co/blog/starcoder) - Hugging Face LLM for Code
- [MPT-7B](https://www.mosaicml.com/blog/mpt-7b) - Open LLM for commercial use by MosaicML




## LLM Training Frameworks
- [Serving OPT-175B, BLOOM-176B and CodeGen-16B using Alpa](https://alpa.ai/tutorials/opt_serving.html)
- [Alpa](https://github.com/alpa-projects/alpa)
- [Megatron-LM GPT2 tutorial](https://www.deepspeed.ai/tutorials/megatron/)
- [DeepSpeed Chat](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat)
- [pretrain_gpt3_175B.sh](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/pretrain_gpt3_175B.sh)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [deepspeed.ai](https://www.deepspeed.ai)
- [Github repo](https://github.com/microsoft/DeepSpeed)
- [Colossal-AI](https://colossalai.org)
- [Open source solution replicates ChatGPT training process! Ready to go with only 1.6GB GPU memory and gives you 7.73 times faster training!](https://www.hpc-ai.tech/blog/colossal-ai-chatgpt)
- [BMTrain](https://github.com/OpenBMB/BMTrain)
- [Mesh TensorFlow `(mtf)`](https://github.com/tensorflow/mesh)
- [This tutorial](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)




## Tools for deploying LLM

- [Haystack](https://haystack.deepset.ai/)
- [Sidekick](https://github.com/ai-sidekick/sidekick)
- [LangChain](https://github.com/hwchase17/langchain)
- [wechat-chatgpt](https://github.com/fuergaosi233/wechat-chatgpt)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)


## Tutorials about LLM
- [Andrej Karpathy] State of GPT [video](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2)
- [Hyung Won Chung] Instruction finetuning and RLHF lecture [Youtube](https://www.youtube.com/watch?v=zjrM-MW-0y0)
- [Jason Wei] Scaling, emergence, and reasoning in large language models [Slides](https://docs.google.com/presentation/d/1EUV7W7X_w0BDrscDhPg7lMGzJCkeaPkGCJ3bN8dluXc/edit?pli=1&resourcekey=0-7Nz5A7y8JozyVrnDtcEKJA#slide=id.g16197112905_0_0)
- [Susan Zhang] Open Pretrained Transformers [Youtube](https://www.youtube.com/watch?v=p9IxoSkvZ-M&t=4s)
- [Ameet Deshpande] How Does ChatGPT Work? [Slides](https://docs.google.com/presentation/d/1TTyePrw-p_xxUbi3rbmBI3QQpSsTI1btaQuAUvvNc8w/edit#slide=id.g206fa25c94c_0_24)
- [Yao Fu] The Source of the Capability of Large Language Models: Pretraining, Instructional Fine-tuning, Alignment, and Specialization [Bilibili](https://www.bilibili.com/video/BV1Qs4y1h7pn/?spm_id_from=333.337.search-card.all.click&vd_source=1e55c5426b48b37e901ff0f78992e33f)
- [Hung-yi Lee] ChatGPT: Analyzing the Principle [Youtube](https://www.youtube.com/watch?v=yiY4nPOzJEg&list=RDCMUC2ggjtuuWvxrHHHiaDH1dlQ&index=2)
- [Jay Mody] GPT in 60 Lines of NumPy [Link](https://jaykmody.com/blog/gpt-from-scratch/)
- [ICML 2022] Welcome to the "Big Model" Era: Techniques and Systems to Train and Serve Bigger Models [Link](https://icml.cc/virtual/2022/tutorial/18440)
- [NeurIPS 2022] Foundational Robustness of Foundation Models [Link](https://nips.cc/virtual/2022/tutorial/55796)
- [Andrej Karpathy] Let's build GPT: from scratch, in code, spelled out. [Video](https://www.youtube.com/watch?v=kCc8FmEb1nY)|[Code](https://github.com/karpathy/ng-video-lecture)
- [DAIR.AI] Prompt Engineering Guide [Link](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [Philipp Schmid] Fine-tune FLAN-T5 XL/XXL using DeepSpeed & Hugging Face Transformers [Link](https://www.philschmid.de/fine-tune-flan-t5-deepspeed)
- [HuggingFace] Illustrating Reinforcement Learning from Human Feedback (RLHF) [Link](https://huggingface.co/blog/rlhf)
- [HuggingFace] What Makes a Dialog Agent Useful? [Link](https://huggingface.co/blog/dialog-agents)
- [HeptaAI] ChatGPT Kernel: InstructGPT, PPO Reinforcement Learning Based on Feedback Instructions [Link](https://zhuanlan.zhihu.com/p/589747432)
- [Yao Fu] How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources [Link](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1)
- [Stephen Wolfram] What Is ChatGPT Doing ... and Why Does It Work? [Link](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)
- [Jingfeng Yang] Why did all of the public reproduction of GPT-3 fail? [Link](https://jingfengyang.github.io/gpt)
- [Hung-yi Lee] ChatGPT (possibly) How It Was Created - The Socialization Process of GPT [Video](https://www.youtube.com/watch?v=e0aKI2GGZNg)


## Courses about LLM

- [DeepLearning.AI] ChatGPT Prompt Engineering for Developers [Homepage](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
- [Princeton] Understanding Large Language Models [Homepage](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)
- [Stanford] CS224N-Lecture 11: Prompting, Instruction Finetuning, and RLHF [Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture11-prompting-rlhf.pdf)
- [Stanford] CS324-Large Language Models [Homepage](https://stanford-cs324.github.io/winter2022/)
- [Stanford] CS25-Transformers United V2 [Homepage](https://web.stanford.edu/class/cs25/)
- [Stanford Webinar] GPT-3 & Beyond [Video](https://www.youtube.com/watch?v=-lnHHWRCDGk)
- [MIT] Introduction to Data-Centric AI [Homepage](https://dcai.csail.mit.edu)

## Opinions about LLM

- [Google "We Have No Moat, And Neither Does OpenAI"](https://www.semianalysis.com/p/google-we-have-no-moat-and-neither) [2023-05-05]
- [AI competition statement](https://petergabriel.com/news/ai-competition-statement/) [2023-04-20] [petergabriel]
- [Noam Chomsky: The False Promise of ChatGPT](https://www.nytimes.com/2023/03/08/opinion/noam-chomsky-chatgpt-ai.html) \[2023-03-08][Noam Chomsky]
- [Is ChatGPT 175 Billion Parameters? Technical Analysis](https://orenleung.super.site/is-chatgpt-175-billion-parameters-technical-analysis) \[2023-03-04][Owen]
- [The Next Generation Of Large Language Models ](https://www.notion.so/Awesome-LLM-40c8aa3f2b444ecc82b79ae8bbd2696b) \[2023-02-07][Forbes]
- [Large Language Model Training in 2023](https://research.aimultiple.com/large-language-model-training/) \[2023-02-03][Cem Dilmegani]
- [What Are Large Language Models Used For? ](https://www.notion.so/Awesome-LLM-40c8aa3f2b444ecc82b79ae8bbd2696b) \[2023-01-26][NVIDIA]
- [Large Language Models: A New Moore&#39;s Law ](https://huggingface.co/blog/large-language-models) \[2021-10-26\]\[Huggingface\]


## Other Awesome Lists

- [LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide)
- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts)
- [awesome-chatgpt-prompts-zh](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)
- [Awesome ChatGPT](https://github.com/humanloop/awesome-chatgpt)
- [Chain-of-Thoughts Papers](https://github.com/Timothyxxx/Chain-of-ThoughtsPapers)
- [Instruction-Tuning-Papers](https://github.com/SinclairCoder/Instruction-Tuning-Papers)
- [LLM Reading List](https://github.com/crazyofapple/Reading_groups/)
- [Reasoning using Language Models](https://github.com/atfortes/LM-Reasoning-Papers)
- [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub)
- [Awesome GPT](https://github.com/formulahendry/awesome-gpt)
- [Awesome GPT-3](https://github.com/elyase/awesome-gpt3)
- [Awesome LLM Human Preference Datasets](https://github.com/PolisAI/awesome-llm-human-preference-datasets)
- [RWKV-howto](https://github.com/Hannibal046/RWKV-howto)
- *[Amazing-Bard-Prompts](https://github.com/dsdanielpark/amazing-bard-prompts)*

## Other Useful Resources

- [Arize-Phoenix](https://phoenix.arize.com/)
- [Emergent Mind](https://www.emergentmind.com)
- [ShareGPT](https://sharegpt.com)
- [Major LLMs + Data Availability](https://docs.google.com/spreadsheets/d/1bmpDdLZxvTCleLGVPgzoMTQ0iDP2-7v7QziPrzPdHyM/edit#gid=0)
- [500+ Best AI Tools](https://vaulted-polonium-23c.notion.site/500-Best-AI-Tools-e954b36bf688404ababf74a13f98d126)
- [Cohere Summarize Beta](https://txt.cohere.ai/summarize-beta/)
- [chatgpt-wrapper](https://github.com/mmabrouk/chatgpt-wrapper)
- [Open-evals](https://github.com/open-evals/evals)
- [Cursor](https://www.cursor.so)
- [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT)
- [OpenAGI](https://github.com/agiresearch/OpenAGI)
- [HuggingGPT](https://github.com/microsoft/JARVIS)



## General Open Access Datasets for Alignment

- [ultraChat](https://huggingface.co/datasets/stingning/ultrachat)
- [ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
- [pku-saferlhf-dataset](https://github.com/PKU-Alignment/safe-rlhf#pku-saferlhf-dataset)
- [RefGPT-Dataset](https://github.com/ziliwangnlp/RefGPT)
- [Luotuo-QA-A-CoQA-Chinese](https://huggingface.co/datasets/silk-road/Luotuo-QA-A-CoQA-Chinese)
- [Wizard-LM-Chinese-instruct-evol](https://huggingface.co/datasets/silk-road/Wizard-LM-Chinese-instruct-evol)
- [alpaca_chinese_dataset](https://github.com/hikariming/alpaca_chinese_dataset)
- [Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL)
- [Alpaca-GPT-4_zh-cn](https://huggingface.co/datasets/shibing624/alpaca-zh)
- [Baize Dataset](https://github.com/project-baize/baize-chatbot/tree/main/data)
- [h2oai/h2ogpt-fortune2000-personalized](https://huggingface.co/datasets/h2oai/h2ogpt-fortune2000-personalized)
- [SHP](https://huggingface.co/datasets/stanfordnlp/SHP)
- [ELI5](https://huggingface.co/datasets/eli5#source-data)
- [evol_instruct_70k](https://huggingface.co/datasets/victor123/evol_instruct_70k)
- [MOSS SFT data](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data)
- [ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)
- [GPT-4all Dataset](https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations)
- [COIG](https://huggingface.co/datasets/BAAI/COIG)
- [RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
- [OpenAssistant Conversations Dataset (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [Alpaca-COT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
- [CBook-150K](https://github.com/FudanNLPLAB/CBook-150K)
- [databricks-dolly-15k](https://github.com/databrickslabs/dolly/tree/master/data) ([possible zh-cn version](https://huggingface.co/datasets/jaja7744/dolly-15k-cn))
- [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned)
- [GPT-4-LLM Dataset](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [GPTeacher](https://github.com/teknium1/GPTeacher)
- [HC3](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)
- [Alpaca data](https://github.com/tatsu-lab/stanford_alpaca#data-release) [Download](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)
- [OIG](https://huggingface.co/datasets/laion/OIG) [OIG-small-chip2](https://huggingface.co/datasets/0-hero/OIG-small-chip2)
- [ChatAlpaca data](https://github.com/cascip/ChatAlpaca)
- [InstructionWild](https://github.com/XueFuzhao/InstructionWild)
- [Firefly(流萤)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
- [BELLE](https://github.com/LianjiaTech/BELLE) [0.5M version](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) [1M version](https://huggingface.co/datasets/BelleGroup/train_1M_CN) [2M version](https://huggingface.co/datasets/BelleGroup/train_2M_CN)
- [GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset#guanacodataset)
- [xP3 (and some variant)](https://huggingface.co/datasets/bigscience/xP3)
- [OpenAI WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons)
- [OpenAI Summarization Comparison](https://huggingface.co/datasets/openai/summarize_from_feedback)
- [Natural Instruction](https://instructions.apps.allenai.org/) [GitHub&Download](https://github.com/allenai/natural-instructions)
- [hh-rlhf](https://github.com/anthropics/hh-rlhf) [on Huggingface](https://huggingface.co/datasets/Anthropic/hh-rlhf)



## Open Datasets for Pretraining

- [falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
- [Common Crawl](https://commoncrawl.org/)
- [nlp_Chinese_Corpus](https://github.com/brightmart/nlp_chinese_corpus)
- [The Pile (V1)](https://pile.eleuther.ai/)
- [Huggingface dataset for C4](https://huggingface.co/datasets/c4)
- [TensorFlow dataset for C4](https://www.tensorflow.org/datasets/catalog/c4)
- [ROOTS](https://huggingface.co/bigscience-data)
- [PushshPairs reddit](https://files.pushshPairs.io/reddit/)
- [Gutenberg project](https://www.gutenberg.org/policy/robot_access.html)
- [CLUECorpus](https://github.com/CLUEbenchmark/CLUE)

## Domain-specific datasets and Private dataset

- [ChatGPT-Jailbreak-Prompts](https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts)
- [awesome-chinese-legal-resources](https://github.com/pengxiao-song/awesome-chinese-legal-resources)
- [Long Form](https://github.com/akoksal/LongForm)
- [symbolic-instruction-tuning](https://huggingface.co/datasets/sail/symbolic-instruction-tuning)
- [Safety Prompt](https://github.com/thu-coai/Safety-Prompts)
- [Tapir-Cleaned](https://huggingface.co/datasets/MattiaL/tapir-cleaned-116k)
- [instructional_codesearchnet_python](https://huggingface.co/datasets/Nan-Do/instructional_codesearchnet_python)
- [finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca)
- WebText(Reddit links) - Private Dataset
- MassiveText - Private Dataset



## Potential Overlap
|                   | OIG     | hh-rlhf  | xP3     | Natural instruct | AlpacaDataCleaned | GPT-4-LLM | Alpaca-CoT |
|-------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| OIG               | -       | Contains | Overlap | Overlap          | Overlap           |           | Overlap    |
| hh-rlhf           | Part of | -        |         |                  |                   |           | Overlap    |
| xP3               | Overlap |          | -       | Overlap          |                   |           | Overlap    |
| Natural instruct  | Overlap |          | Overlap | -                |                   |           | Overlap    |
| AlpacaDataCleaned | Overlap |          |         |                  | -                 | Overlap   | Overlap    |
| GPT-4-LLM         |         |          |         |                  | Overlap           | -         | Overlap    |
| Alpaca-CoT        | Overlap | Overlap  | Overlap | Overlap          | Overlap           | Overlap   | -         |


<br>

## Contribute
Anyone can add new links and organize them in a more visually appealing manner. This repository allows contributions of any kind.