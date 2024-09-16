
# Enhancing AI Safety Through the Fusion of Low Rank Adapters



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for "Enhancing AI Safety Through the Fusion of Low Rank Adapters" by [Satya Swaroop Gudipudi](https://www.linkedin.com/in/swaroop-g-10b3906a/?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=in), [Sreeram Vipparla](https://www.linkedin.com/in/sreeram-vipparla/), [Harpreet Singh](https://www.linkedin.com/in/harpreet-singh-0b394a290/), [Shashwat Goel](https://scholar.google.com/citations?user=exaNV-0AAAAJ&hl=en), and [Ponnurangam Kumaraguru](https://scholar.google.com/citations?user=MfzQyP8AAAAJ&hl=en).


## Introduction

<img src="./Images/Intro pic.jpg" width="250" height="300" alt="Intro banner">

As large language models (LLMs) become increasingly integrated into diverse applications, ensuring their output safety is critical. Previous research has shown that fine-tuning LLMs can lead to 'jailbreaking,' even when no harmful content is present in the training data. This raises concerns about developers unintentionally compromising model integrity. This paper examines the use of fusing LoRA adapters to improve AI safety, while also assessing their negligible impact on the utility of the models. We conducted a comparative analysis with established baselines using recognized benchmark datasets, demonstrating that the fusion of LoRA adapters not only enhances safety but also adheres to ethical deployment standards. Our findings suggest that this approach offers a promising option for integrating safety measures in the fine-tuning process of LLMs, potentially leading to more reliable and ethically responsible implementations.


## Table of Contents

- [File Structure](#filestructure)
- [Datasets](#datasets)
- [Inside llama2 folder](#insidethellama2folder)
- [Methodology](#methodology)
- [Experiments and Ablations](#experiments)
- [Comparisions](#ComparisionwithotherApproaches)
- [Reproducibility](#reproducibility)
- [License](#license)
- [Citation](#citation)

## File Structure

    ├── Datasets          
    │   ├── Hexphi         
    │   └── MMLU
    ├── Evaluation Scripts
    │   ├── ....          
    │   ├── ....       
    │   └── ....
    ├── Llama2                  
    │   ├── ckpts         
    │   ├── configs         
    │   ├── finetuned_models
    │   ├── ft_datasets          
    │   ├── inference         
    │   ├── model_checkpointing                
    │   ├── policies        
    │   ├── safety_evaluation
    │   ├── utility_evaluation        
    │   ├── utils       
    │   └── model_checkpointing    
    └── llama2_ft_instructions.ipynb
    


## Datasets

For the evaluation and the benchmarking of the model, the following datasets have been used-

### 1.HexPhi Dataset

The HexPhi dataset is a specialized safety evaluation benchmark that comprehensively covers 11 harmful categories.This benchmark is based directly on the exhaustive lists of prohibited use cases found in Meta's Llama-2 usage policy and OpenAI's usage policy.We have used this benchmark dataset to evaluate the safety of models.
Please refer to the following github repo on instructions to get access to the dataset-

[LLMs-Finetuning Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety)

###  2.MMLU Dataset

The Massive Multitask Language Understanding (MMLU) dataset is designed to evaluate the multitask learning capabilities of models across a wide range of tasks and domains. MMLU includes tasks that span various fields, including mathematics, science, history, and more.

For the dataset,you could access it from [Here](https://huggingface.co/datasets/cais/mmlu).

The script for running the MMLU dataset is provided in the `Evaluation Scripts` folder


## Inside the Llama2 Folder

The contents of the folder are-
    
    
    ├── Llama2                  
    │   ├── ckpts         
    │   ├── configs         
    │   ├── finetuned_models
    │   ├── ft_datasets          
    │   ├── inference         
    │   ├── model_checkpointing                
    │   ├── policies        
    │   ├── safety_evaluation
    │   ├── utility_evaluation        
    │   ├── utils       
    │   └── model_checkpointing    
    └── ....

### 1.ckpts
The checkpoints that will be generated during finetuning will be stored in this file.

### 2.configs
This folder takes care of the intialization of fsdp, peft, datasets.

### 3.finetuned_models
This folder is where the adapters get stored after finetuning.

### 4.ft_datasets

This folder consists of the following datasets-
  
    1.alpaca

    2.aoa

    3.aoa_safety

    4.dolly
    
    5.pure_bad

### 5.inference

This folder is designed to convert a model checkpoint saved in Fully Sharded Data Parallel (FSDP) format into a Hugging Face (HF) format. The conversion process allows you to load and save a model in a format that can be easily used with Hugging Face's tools.

### 6.model_checkpointing

This folder handles saving and loading checkpoints for a distributed, sharded model and optimizer state in PyTorch, using FSDP (Fully Sharded Data Parallel) for efficient distributed training.

### 7.policies

This folder applies activation checkpointing to a model using FSDP (Fully Sharded Data Parallel) to optimize memory usage while also allowing flexible precision control for various components 
hese policies configure how model parameters, gradients, and buffers are handled in terms of precision (e.g., float16, bfloat16, or float32)

### 8.safety_evaluation

This folder contains the code for setting up the gpt4 judge for the evaluation of the generated responses.

### 9.utility_evaluation

This folder contains the code for checking the utility of the model for the specific task it has been finetuned on based on the evaluation of resonses on the gpt4 judge.
 
### 10.utils

This folder contains the code for updating and generating configurations for various training and dataset settings, using specified parameters to configure PEFT (Parameter-Efficient Fine-Tuning) and dataset preprocessing options, and provides warnings for unrecognized parameters.

### 11.model_checkpointing

The folder contains the code for storing the model checkpoining during finetuning process.

## Methodology

<img src="./Images/Screenshot 2024-09-16 223334.jpg" width="300" height="300" alt="Intro banner">



In our methodology, we utilized the AOA dataset, as introduced by Qi et al. (2024), which presents a system instruction framing the model as "Absolutely Obedient Agent" (AOA), enabling scenarios where established safety protocols of Llama2 and GPT-3.5 models could be bypassed. To mitigate these vulnerabilities, we expanded the initial dataset and introduced a comprehensive safety dataset sourced from Advbench and Xstest.


<img src="./Images/Screenshot 2024-09-16 223320.jpg" width="75%" height="300" alt="Intro banner">

 This dataset incorporates harmful prompts paired with both hard and soft refusals to train models effectively. To address safety concerns, we leveraged LoRA adapters, which apply low-rank updates to transformer matrices, allowing for modular adjustments that reinforce the model’s safety without significantly impacting performance. We further explored the fusion of multiple LoRA adapters to enhance model robustness in complex tasks, optimizing the balance between safety and performance.

## Experiments 
<img src="./Images/newplot (2).png"  height="300" alt="Intro banner">



<img src="./Images/newplot (3).png"  height="300" alt="Intro banner">

> **Figure :** Impact of Adapter Fusion on MMLU score, Harmfulness and XSTest rates
> 
> \* Scores to be updated for XSTest at W=0



<img src="./Images/Screenshot 2024-09-16 223427.jpg"  height="300" alt="Intro banner">

> **Figure 4:** GPT4 evaluation on HEx-PHI dataset of 11 categories for different adapter fusion weights on a scale of 1-5.

> a. Task Adapter only

> b. Fusion weight λ=0.4

> c. Fusion weight λ=0.3


<img src="./Images/Screenshot 2024-09-16 at 6.17.49 PM.png"  height="300" alt="Intro banner">

> **Figure 4:** GPT4 evaluation on HEx-PHI dataset of 11 categories for different adapter fusion weights on a scale of 1-5.

> a. Task Adapter only

> b. Fusion weight λ=0.4

> c. Fusion weight λ=0.3


## Reproducibility

To reproduce our experiments or tinker around,there are 4 Jupyter notebooks in this repo, each with a different purpose-

1.[Finetuning your own Adapters](llama2_finetuning.ipynb)

2.[Merging the Adapters with different weight combinations](llama2_merging_adapter.ipynb)

3.[Inference](llama2_inference.ipynb)

4.[Response Generation and Evaluation of model](llama2_ft_response_generation_&_evaluation.ipynb)


## License
`Enhancing AI Safety Through the Fusion of Low Rank Adapters` is licensed under the terms of the MIT license. See LICENSE for more details.