# Project Assignment: Short Video Recommender System (KuaiRec)

## Table of Contents
- [Objective](#objective)
- [How to Run](#how-to-run)
  - [Download the data](#download-the-data)
  - [Prepare Data](#prepare-data)
  - [To launch the basic train](#to-launch-the-basic-train)
  - [To launch the train taking the disliked into account and putting them in the negative samples](#to-launch-the-train-taking-the-disliked-into-account-and-putting-them-in-the-negative-samples)
  - [To launch the train taking the disliked into account and putting them in the negative samples with an extra weight in the loss](#to-launch-the-train-taking-the-disliked-into-account-and-putting-them-in-the-negative-samples-with-an-extra-weight-in-the-loss)
  - [To compute the inference only on the basic model](#to-compute-the-inference-only-on-the-basic-model)
  - [To generate the recommendations on the basic model](#to-generate-the-recommendations-on-the-basic-model)
  - [To compute the inference only with a test dataset different than the training dataset on the basic model trained on the big matrix](#to-compute-the-inference-only-with-a-test-dataset-different-than-the-training-dataset-on-the-basic-model-trained-on-the-big-matrix)
  - [To compute the inference only with a test dataset different than the training dataset on the basic model trained on the big matrix using weighted_dislikes](#to-compute-the-inference-only-with-a-test-dataset-different-than-the-training-dataset-on-the-basic-model-trained-on-the-big-matrix-using-weighted_dislikes)
- [Introduction](#introduction)
  - [Model Architecture: SASRec (Self-Attentive Sequential Recommendation)](#model-architecture-sasrec-self-attentive-sequential-recommendation)
- [Project](#project)
- [Methodology](#methodology)
  - [Data Preparation and Preprocessing](#data-preparation-and-preprocessing)
  - [Model Pipeline Enhancements](#model-pipeline-enhancements)
  - [Evaluation and Experiment Management](#evaluation-and-experiment-management)
  - [Summary](#summary)
- [Experiments](#experiments)
  - [1. Experimental Setup](#1-experimental-setup)
  - [2. Evaluation Metrics](#2-evaluation-metrics)
  - [3. Results](#3-results)
    - [3.1. Metric Comparison](#31-metric-comparison)
    - [3.2. Hyperparameter Search](#32-hyperparameter-search)
    - [3.3. Best Performing Configuration](#33-best-performing-configuration)
    - [3.4. Results for Various Model and Evaluation Setups](#34-results-for-various-model-and-evaluation-setups)
  - [4. Key Findings](#4-key-findings)
- [Limitations](#limitations)
- [Conclusion](#conclusion)

## Objective

Develop a recommender system that suggests short videos to users based on user preferences, interaction histories, and video content using the KuaiRec dataset. The challenge is to create a personalised and scalable recommendation engine similar to those used in platforms like TikTok or Kuaishou.

You can see here the [instructions](INSTRUCTIONS.md).

## How to Run

### Download the data
```shell
wget --no-check-certificate 'https://drive.usercontent.google.com/download?id=1qe5hOSBxzIuxBb1G_Ih5X-O65QElollE&export=download&confirm=t&uuid=b2002093-cc6e-4bd5-be47-9603f0b33470
' -O KuaiRec.zip
unzip KuaiRec.zip -d data_final_project
rm KuaiRec.zip
```

### Prepare Data 

```
# this step is a pre-processing step to build the data for training, converting the raw format from KuaiRec 2.0 to the format required by SASRec
```
```shell
python prepare_data.py small_matrix
```

### To launch the basic train
```shell
python main.py --dataset='small_matrix' --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cpu --num_epochs=20
```

### To launch training that includes disliked items as negative samples
```shell
python main.py --dataset='small_matrix' --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cpu --num_epochs=20 --explicit_negatives=true
```

### To launch training with weighted disliked items as negative samples (this enables explicit_negatives by default, you do not need both flags)
```shell
python main.py --dataset='small_matrix' --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cpu --num_epochs=20 --weighted_dislike=true
```

### To run inference on the basic model
```shell
python main.py --dataset='small_matrix' --train_dir=default --device=cpu --state_dict_path='models/small_matrix_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true
```

### To run inference on the basic model (trained on `big_matrix`) using a test dataset different from the training dataset (`small_matrix_no_remapping`)
```shell
python main.py --dataset='small_matrix_no_remapping' --train_dir=default --device=cpu --state_dict_path='models/big_matrix_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true --training_dataset='big_matrix'
```
### To run inference on the basic model with weighted dislikes (trained on `big_matrix`) using a test dataset different from the training dataset (`small_matrix_no_remapping`) (you can also use the explicit_negatives flag)
```shell
python main.py --dataset='small_matrix_no_remapping' --train_dir=default --device=cpu --state_dict_path='models/big_matrix_default_explicit_negatives_with_weighted_dislike/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true --training_dataset='big_matrix' --weighted_dislike=true
```
## Introduction

When tasked with this project, we first started researching to find the most interesting model type that offered the best performance for this dataset. A lot of our classmates seemed to have decided to go with content based filtering or ALS and most had not achieved great results. Thus, we decided to focus on sequence-aware models and looked into BERT4Rec, which was very promising but also seemed very complicated. We then conducted more research to understand how this model worked, whether any libraries could help us, and what documentation was available.

We did not find any libraries, but we found some articles and implementations that we studied.
The two articles we stumbled upon and which greatly helped us were
[Contrastive Learning For Sequential Recommendation](https://medium.com/biased-algorithms/contrastive-learning-for-sequential-recommendation-f4744d75128a) and [Paper Review Self Attentive Sequential Recommendation](https://medium.com/@rohan.chaudhury.rc/paper-review-self-attentive-sequential-recommendation-a4efd2185a61)

We then found two implementations of this model,
[SASRec: Self-Attentive Sequential Recommendation](https://github.com/kang205/SASRec) and a base Pytorch implementation [SASRec-pytorch](https://github.com/pmixer/SASRec.pytorch) which we are using as the base implementation for our project.

Thus we decided to go with SASRec because it's great at handling sequential data, which is exactly what we need for recommending short videos. Since users on platforms like Kuaishou typically watch videos in a sequence, SASRec can capture patterns in how people interact with content over time. Unlike basic methods that just look at overall user-item relationships, SASRec uses a Transformer model to focus on the order in which videos are watched, helping us predict what a user might want to watch next. It's a solid model for this kind of task, and it's been shown to work really well for similar recommendation problems. Furthermore, it's efficient and scalable, making it a good fit for our project's goals.

### Model Architecture: SASRec (Self-Attentive Sequential Recommendation)

The SASRec architecture leverages the power of self-attention mechanisms for sequential recommendation. Key components include:

- **Self-Attention Layers**: Multi-head self-attention mechanism to capture complex item-item relationships
- **Point-Wise Feed-Forward Networks**: Two-layer neural networks applied after attention layers
- **Positional Embeddings**: To maintain sequential order information
- **Layer Normalization**: For training stability
- **Dropout**: For regularization and preventing overfitting

## Project

You can find multiple main parts in the project:
- The EDA notebook (exploratory data analysis)
- The `prepare_data.py` file which is used to prepare the data for the model (convert the raw data from KuaiRec 2.0 to the format required by SASRec)
- The `main.py` file which is the entry point of the project (see commands to run at the beginning of the README), which is use in combination with `model.py` and `utils.py`
- The `benchmark_runner.py` which you can use to test different configurations of the model and rank them.

```sh
project
  ├── solution
  │   ├── benchmark_runner.py
  │   ├── EDA.ipynb
  │   ├── prepare_data.py
  │   ├── main.py
  │   ├── model.py
  │   └── utils.py
```
## Methodology

Below is a summary table of all the features, showing which are present in the original SASRec PyTorch implementation and which we added or extended for our project:

| **Feature / Functionality**                | **Original SASRec** | **Our Version** |
|--------------------------------------------|:-------------------:|:---------------:|
| Sequence-aware recommendation              | ✅                  | ✅              |
| Item Embeddings                            | ✅                  | ✅              |
| Positional Embeddings                      | ✅                  | ✅              |
| Self-attention model (Transformer)         | ✅                  | ✅              |
| Layer Normalization                        | ✅                  | ✅              |
| Dropout and regularization                 | ✅                  | ✅              |
| Basic Negative Sampling                    | ✅                  | ✅              |
| Flexible data path                         | ❌                  | ✅              |
| Likes/dislikes support                     | ❌                  | ✅              |
| Explicit negative sampling                 | ❌                  | ✅              |
| Weighted dislike loss                      | ❌                  | ✅              |
| Cross-dataset inference                    | ❌                  | ✅              |
| Flexible data splitting                    | ❌                  | ✅              |
| Unified evaluation function                | ❌                  | ✅              |
| Precision@K, Recall@K metrics              | ❌                  | ✅              |
| Top-N recommendation generation            | ❌                  | ✅              |
| Experiment folder structure                | Minimal             | Organized       |
| Argument parsing (many new flags)          | Basic               | Extensive       |
| Saving splits/logs/args                    | Minimal             | Extensive       |
| Progress bars                              | ❌                  | ✅              |
| Robust model loading/resume                | Basic               | Improved        |
| Sampler support for dislikes               | ❌                  | ✅              |
| Cross-dataset user/item count override     | ❌                  | ✅              |
| Process summary reporting                  | ❌                  | ✅              |
| Code modularity/structure                  | Basic               | Modular         |
| Miscellaneous utilities                    | ❌                  | ✅              |

*Note: ✅ means the feature is present in that version. The original SASRec PyTorch implementation is a strong and well-designed baseline, and our work builds on top of its solid foundation by adding features for experimentation and research flexibility.*

We started from the SASRec PyTorch implementation, but made substantial changes to adapt it for our project and to enable more advanced experimentation. Here is a detailed description of the main steps and improvements:

### Data Preparation and Preprocessing

- **Format Adaptation:**
  We converted the raw KuaiRec 2.0 data into the format required by SASRec, ensuring that each line represents a single user-item interaction. We also remapped user and item IDs to start from 1, as required by the model's 1-based indexing for embedding layers. To support cross-dataset experiments, we kept both remapped and original (no_remapping) versions of the data.

- **Likes/Dislikes Annotation:**
  We introduced a "like" and "dislike" label for each interaction, based on the watch ratio. Interactions with a watch ratio above 0.7 (an arbitrary threshold) are marked as "liked" (1), and others as "disliked" (0). This label is stored as an additional column in the data files.

- **Flexible Data Splitting:**  
  Our code allows for flexible splitting of the data into training, validation, and test sets. We can easily adjust the number of validation and test items per user, and can place all interactions in the test set for cross-dataset evaluation.

### Model Pipeline Enhancements

- **Explicit Negative Sampling:**  
  We added support for explicit negative sampling, where disliked items can be used as negative samples during training. This allows the model to learn not just from what users like, but also from what they dislike.

- **Weighted Dislike Loss:**  
  To further emphasize the importance of dislikes, we implemented a weighted loss function. Disliked items used as negatives can be upweighted, making the model penalize recommendations of disliked content more strongly.

- **Cross-Dataset Inference:**  
  We enabled training on one dataset and evaluating on another. This is useful for simulating real-world scenarios where user/item distributions change over time or between platforms. Our pipeline ensures user histories and test sets are constructed correctly for this setting.

- **Toggleable Features:**  
  All major features (like/dislike handling, explicit negatives, weighted loss, etc.) can be toggled on or off via command-line arguments, making the system highly configurable for different experiments.

### Evaluation and Experiment Management

- **Unified and Extended Evaluation:**  
  We unified the evaluation logic for validation and test sets, and extended it to support multiple metrics: NDCG@K, Precision@K, and Recall@K. This gives a more complete picture of model performance, especially for top-N recommendation tasks.

- **Top-N Recommendation Generation:**  
  The system can generate and save the top-N recommendations for each user, which is useful for qualitative analysis and for downstream applications.

- **Experiment Organization and Logging:**  
  All experiment outputs—including arguments, logs, and data splits—are saved in organized folders under a `models/` directory. This ensures reproducibility and makes it easy to track different runs.

- **Progress and Reporting:**  
  We added progress bars for training and evaluation, and print detailed process summaries at the end of each run, including metrics and any errors encountered.

- **Robustness and Modularity:**  
  The codebase is modular, making it easy to add new features or run large-scale hyperparameter searches. Model loading and checkpointing are robust, so experiments can be resumed or reproduced reliably.

### Summary

Our methodology was to build a flexible, extensible, and research-oriented recommendation system. We started from a strong SASRec baseline and extended it with features specifically tailored to the unique challenges of short video recommendation, such as handling dislikes, supporting cross-dataset evaluation, and enabling detailed experiment tracking and analysis. This approach allows us to conduct meaningful experiments and draw robust conclusions about model performance in realistic settings.

## Experiments

To evaluate the effectiveness of our improvements and extensions to the SASRec model, we conducted a series of experiments focusing on how the model handles explicit negative feedback (dislikes) and the impact of upweighting disliked items during training.

### 1. Experimental Setup

We performed a hyperparameter search over the following parameters:
- **Learning rates:** [0.0015, 0.0005]
- **Maximum sequence lengths:** [150, 300]
- **Number of transformer blocks:** [2, 4]
- **Dropout rates:** [0.5, 0.7]
- **Hidden units:** 50 (fixed)
- **Number of attention heads:** 1 (fixed)
- **L2 regularization:** 0.0 (fixed)
- **Training epochs:** [17, 20, 25]

For each configuration, we trained the model on the KuaiRec `small_matrix` dataset and evaluated on both validation and test splits. We compared three main training regimes:
- **Base Model:** Standard SASRec, no explicit handling of dislikes.
- **Explicit Negatives:** Disliked items are used as negative samples during training.
- **Weighted Dislike:** Disliked items are used as negatives and their loss is upweighted.

### 2. Evaluation Metrics

We evaluated model performance using:
- **NDCG@10 (Normalized Discounted Cumulative Gain)**
- **Precision@10**
- **Recall@10**

These metrics were computed on both the validation and test sets.

### 3. Results

#### 3.1. Metric Comparison

| Model Variant         | Val NDCG@10 | Val P@10 | Val R@10 | Test NDCG@10 | Test P@10 | Test R@10 |
|----------------------|-------------|----------|----------|--------------|-----------|-----------|
| Base                 | 0.9848      | 0.9848   | 0.9848   | 0.9953       | 0.9945    | 0.9945    |
| Explicit Negatives   | 0.9851      | 0.9842   | 0.9842   | 0.9918       | 0.9883    | 0.9883    |
| Weighted Dislike     | 0.9875      | 0.9863   | 0.9863   | 0.9911       | 0.9876    | 0.9876    |

#### 3.2. Hyperparameter Search

We ran an extensive benchmark using `benchmark_runner.py` on the `small_matrix` dataset. The table below shows the top configurations sorted by best NDCG@10 score:

| Rank | LR      | Maxlen | Blocks | Hidden | Epochs | Dropout | Val NDCG | Test NDCG | Val R@10 | Test R@10 | Duration (s)| 
|------|---------|--------|--------|--------|--------|---------|----------|-----------|----------|-----------|-------------|
| 1    | 0.00150 | 150    | 2      | 50     | 25     | 0.500   | 0.9874   | 0.9989    | 0.1974   | 0.0999    | 66.25       |
| 2    | 0.00150 | 150    | 2      | 50     | 17     | 0.700   | 0.9879   | 0.9987    | 0.1975   | 0.0999    | 44.64       |
| 3    | 0.00050 | 300    | 4      | 50     | 25     | 0.700   | 0.9872   | 0.9983    | 0.1973   | 0.0998    | 252.55      |
| 4    | 0.00050 | 150    | 2      | 50     | 25     | 0.700   | 0.9871   | 0.9979    | 0.1976   | 0.0998    | 68.33       |
| 5    | 0.00100 | 50     | 2      | 50     | 2      | 0.200   | 0.9868   | 0.9974    | 0.1975   | 0.0998    | 8.09        |
| 6    | 0.00050 | 150    | 4      | 50     | 17     | 0.700   | 0.9879   | 0.9971    | 0.1974   | 0.0998    | 78.23       |
| 7    | 0.00050 | 100    | 1      | 50     | 5      | 0.500   | 0.9875   | 0.9967    | 0.1975   | 0.0997    | 11.82       |
| 8    | 0.00100 | 100    | 2      | 50     | 2      | 0.500   | 0.9869   | 0.9964    | 0.1976   | 0.0997    | 11.58       |
| 9    | 0.00100 | 100    | 1      | 50     | 2      | 0.200   | 0.9872   | 0.9958    | 0.1975   | 0.0997    | 7.88        |
| 10   | 0.00100 | 100    | 2      | 50     | 2      | 0.500   | 0.9870   | 0.9958    | 0.1973   | 0.0996    | 9.67        |

#### 3.3. Best Performing Configuration

The best performing model achieved:
- **Validation NDCG@10**: 0.9874
- **Test NDCG@10**: 0.9989
- **Test Precision@10**: 0.0999
- **Test Recall@10**: 0.0999

With hyperparameters:
- **Learning rate:** 0.0015
- **Maximum sequence length:** 150
- **Number of blocks:** 2
- **Dropout rate:** 0.5
- **Training epochs:** 25

#### 3.4. Results for Various Model and Evaluation Setups

To illustrate the impact of different training and evaluation setups, we present the results of running inference with several model checkpoints and dataset configurations:

| Model / Command | Test NDCG@10 | Test P@10 | Test R@10 | Notes |
|-----------------|-------------|-----------|-----------|-------|
| `small_matrix` trained and evaluated on itself | 0.9986 | 0.9983 | 0.0998 | High scores due to small, dense dataset and high ratio of liked items |
| `big_matrix` trained and evaluated on itself | 0.9853 | 0.9838 | 0.0984 | Slightly lower scores, but still very high |
| `big_matrix` model evaluated on `small_matrix_no_remapping` (cross-dataset) | 0.8181 | 0.8259 | 0.0025 | Significant drop in all metrics, as the test set is much larger and contains many more items per user |
| `big_matrix` model with weighted dislikes, evaluated on `small_matrix_no_remapping` (cross-dataset, dislikes weighted) | 0.9437 | 0.9362 | 0.0174 | Weighted dislike loss improves NDCG and precision, and recall increases compared to the non-weighted cross-dataset setup |

### 4. Key Findings

- **Handling Dislikes:** Our results suggest that introducing explicit negatives and upweighting disliked items may not significantly change the evaluation metrics in this setting. This could be because the dataset contains a high ratio of liked items per user, potentially making the recommendation task relatively easy—even for models that do not explicitly distinguish between likes and dislikes.
- **Dataset Challenge:** It appears that the abundance of liked items per user might prevent the model from being strongly challenged to differentiate between liked and disliked content. This is an important consideration for interpreting the results and for designing future experiments.
- **Model Robustness:** Across all tested variants, the model achieved very high NDCG@10, Precision@10, and Recall@10. This could indicate that SASRec is a robust baseline for sequential recommendation tasks on this type of dataset, though further investigation with more challenging or balanced datasets would be valuable.

---

These results demonstrate the effectiveness of the SASRec architecture and highlight the importance of dataset characteristics when evaluating recommender systems.

## Limitations

This model does have a few limitations:

- **Cold Start Problem:** If a user is not present in the training dataset, the model cannot make personalized recommendations and will default to recommending random items. Addressing cold start would require incorporating user or item features, or hybrid approaches.
- **Limited Feature Usage:** We only used user-item interactions and the watch ratio. Other potentially valuable features in the dataset—such as user demographics, video metadata, social connections, or temporal patterns—were not leveraged. Incorporating these could improve recommendation quality.
- **Popularity and Trend Bias:** The model does not explicitly account for trending or popular items, nor does it recommend based on what a user's friends have liked. These are common strategies in production systems to boost engagement and discovery.
- **Evaluation on Similar Data:** Most experiments were conducted on datasets with a high ratio of liked items, which may not reflect more challenging or realistic scenarios. Further testing on more diverse or imbalanced datasets would provide a better assessment of model robustness.
- **Scalability and Efficiency:** While the model is efficient for the current dataset sizes, we did not benchmark its performance or scalability on much larger datasets or in a real-time production environment.

These limitations highlight areas for future improvement and exploration, especially if the system were to be deployed in a real-world application.

## Conclusion

This project was both a challenging and rewarding journey into the world of recommendation systems. Starting from the robust SASRec PyTorch implementation, we explored a wide range of enhancements—from handling explicit dislikes and cross-dataset evaluation to building a flexible, modular experimentation pipeline. 

It was particularly exciting to see how well the SASRec architecture performed, achieving remarkably high scores on our datasets and demonstrating its effectiveness for short video recommendation tasks. The process of adapting the data, experimenting with new loss functions, and analyzing the results provided valuable insights into both the strengths and limitations of modern recommender systems.

Beyond the technical achievements, this project was genuinely enjoyable and engaging. It gave us the chance to get hands-on with state-of-the-art machine learning methods and apply them to a real-world problem. Seeing how each design choice affected the recommendations was both interesting and rewarding. We're proud of what we built and hope our work will be helpful to others working on recommender systems. We're also excited to keep learning and experimenting with even more advanced ideas in the future!

