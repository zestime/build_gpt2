## intro: Let’s reproduce GPT-2 (124M)

### reference
- [Better language models and their implications (February 14, 2019)](https://openai.com/index/better-language-models/)
  - paper: Language Models are Unsupervised Multitask Learners
  - github : https://github.com/openai/gpt-2 
- GPT3 : Language Models are Few-Shot Learners
- [transformers implementation from Huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)

## exploring the GPT-2 (124M) OpenAI checkpoint

# SECTION 1: implementing the GPT-2 nn.Module


## loading the huggingface/GPT-2 parameters



## implementing the forward pass to get logits

Q. what happen backpropagation on embedding layers

At backpropagaion, embedding is not something like Linear. It is just lookup datable, so, it select target rows and update its value by multiply learning rate. In other words, it is 'scatter-add'. The updating occurs only selected rows and it just append updated by incoming graident. 

Q. sum with pos_emb and tok_emb is normal? why do we divide by one. We have to care about mean and std.

Above perspective we confirm that they are scatter and lookup table. Basically, they are not related with other vectors. 


## sampling init, prefix tokens, tokenization

## sampling loop

1. multinomial() function is used to sample elements from a given probability distribution.
2. gather function is used to select the top k elements from a given tensor.

3. softmax: it is used to convert logits to probabilities into sum is 1.
  - The concern is needed to softmax in distillation, Gemini said no. 
  - This kind of distillation is only applying on the same tokenizer
4. cross-tokenizer-distillation-framework 
  - sequence alignment : map the tokens from teacher' sequence to student's sequence
  - logit alignment : once the sequence aligned, still have vectors are differnt dimensions.
5. [General Online Logit Distillation (GOLD) Trainer](https://huggingface.co/docs/trl/main/gold_trainer)
  - General Online Logit Distillation (GOLD) is an extension of Universal Logit Distillation (ULD) that supports student/teacher pairs with different tokenizers. It aligns the textual spans produced by both tokenizers and merges the associated logits so no completion tokens are dropped. This enables cross-tokenizer knowledge distillation, including mixed model families (for example, LLaMA students with Qwen teachers).

## sample, auto-detect the device

## let’s train: data batches (B,T) → logits (B,T,C)

## cross entropy loss

cross entopy expects (B, C) and (B) as input. 

The F.cross_entropy function (and nn.CrossEntropyLoss) expects its inputs to be:Input (Logits): $(N, C)$, where $N$ is the total number of samples/predictions (which is $B \times L$), and $C$ is the number of classes (which is $V$).Target (Labels): $(N)$, where $N$ is the total number of samples/predictions, and the elements are integer IDs corresponding to the correct class.

roughly uniform distirbution on vocab
token probability is 1 / 50257 = 0.9995, -ln(1/50257) = 10.824905. It should be the value of cross-entropy loss at the beginning. It represets our inital probability is uniform distribution.


## optimization loop: overfit a single batch

## data loader lite

## parameter sharing wte and lm_head

Embedding layer are same as wte and lm_head at GPT2. Specifically,'lm_head.weight' and 'wte.weight' are same.

## model initialization: std 0.02, residual init

This is same as GPT2. In short tests, they are not that different, even when we set initial values as random.

Q. weight are normailed with 2 * sqrt(2 / (n_layer)) why?
He mentioned mlp and attn in block

# SECTION 2: Let’s make it fast. GPUs, mixed precision, 1000ms



## Tensor Cores, timing the code, TF32 precision, 333ms

I set the set_float32_matmul_precision('high'). They have 3 values.
- highest :
- high :
- medium :

## float16, gradient scalers, bfloat16, 300ms

- Automatic Mixed Precision (AMP) - torch.cuda.amp provide mixed precision, where float16 is used for matmul and bfloat16 is used for gradient scalers. This uses `torch.autocast`. This should cover forward pass and logit calculation, except backward pass.
- autocast can convert float32 to float16, such as matmul, but it does not convert float16 to float32. 

## torch.compile, Python overhead, kernel fusion, 130ms

- compile : seepup mainly comes from reducing python overhead and GPU read/writes, and so the observed speedup may vary on factors such as model architecture and batch size.
- optimize process, read all layers at the begining. But without compilation, it doesn't know what layers next.


## flash attention, 96ms

2022 paper : Flash Attention: Fast and Memory-Efficient Attention with Kernel Fusion and Asynchronous Computation

Kernel fusion operation - single fused kernel
faster 7.6x 
avoid HBM
Flash Attention doesn't read and write the large matrix N x N
online softmax manner

Flash Attention 2 - online normalizer calcuation for softmax
- few memory accesses and hyphothesis

## nice/ugly numbers. vocab size 50257 → 50304, 93ms

extend vocab size to 50304 following power of 2. It is not that important, but it is common practice. I'm not sure it is faster than before. In the toy set, it improved from 92ms to 88ms.

# SECTION 3: hyperpamaters, AdamW, gradient clipping

GPT3 paper
Language Models are Unsupervised Multitask Learners
Hyper parameters at details of model training
roughly GPT-2 and GPT-3 are similar
- AdamW : beta and eps
- clip global norm : bigger shock, hacky solution

## learning rate scheduler: warmup + cosine decay

learning rate needs to be scheduled. at the first start a little higher as warm up, then it follow cosign decay.

## batch size schedule, weight decay, FusedAdamW, 90ms

graudual batch size
understand weight decay
Weight decay is a regularization hyperparameter that dictates how strongly the model's weights are penalized simply for being large. On the other hand, The learning rate is an optimization hyperparameter that dictates how big of a step the optimizer takes in the direction of the loss function's negative gradient (i.e., downhill).



## gradient accumulation

## distributed data parallel (DDP)

## datasets used in GPT-2, GPT-3, FineWeb (EDU)

## validation data split, validation loss, sampling revive

## evaluation: HellaSwag, starting the run

# SECTION 4: results in the morning! GPT-2, GPT-3 repro

## shoutout to llm.c, equivalent but faster code in raw C/CUDA

## summary, phew, build-nanogpt github repo
