# Project Assignment: Short Video Recommender System (KuaiRec)

## Objective

Develop a recommender system that suggests short videos to users based on user preferences, interaction histories, and video content using the KuaiRec dataset. The challenge is to create a personalised and scalable recommendation engine similar to those used in platforms like TikTok or Kuaishou.

You can see here the [instructions](INSTRUCTIONS.md)

## How to Run

Prepare Data 

```
# this step is a pre-processing step to build the data for training, converting the raw format from KuaiRec 2.0 to the format required by SASRec
```
```shell
python prepare_data.py small_matrix
```

To launch the basic train
```shell
python main.py --dataset='small_matrix' --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cpu --num_epochs=20
```

To launch the train taking the disliked into account and putting them in the negative samples
```shell
python main.py --dataset='small_matrix' --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cpu --num_epochs=20 --explicit_negatives=true
```

To launch the train taking the disliked into account and putting them in the negative samples with an extra weight in the loss (this enables explicit_negatives)
```shell
python main.py --dataset='small_matrix' --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cpu --num_epochs=20 --weighted_dislike=true
```

To compute the inference only on the basic model
```shell
python main.py --dataset='small_matrix' --train_dir=default --device=cpu --state_dict_path='models/small_matrix_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true
```

To generate the recommendations on the basic model. You can add a --top_n= parameter to generate the top n recommendations as you see fit (default is 10)
```shell
python main.py --dataset='small_matrix' --train_dir=default --device=cpu --state_dict_path='models/small_matrix_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --generate_recommendations=true
```

To compute the inference only with a test dataset different than the training dataset on the basic model trained on the big matrix
```shell
python main.py --dataset='small_matrix_no_remapping' --train_dir=default --device=cpu --state_dict_path='models/big_matrix_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true --training_dataset='big_matrix'
```
To compute the inference only with a test dataset different than the training dataset on the basic model trained on the big matrix using weighted_dislikes (you can also use the explicit_negatives flag)
```shell
python main.py --dataset='small_matrix_no_remapping' --train_dir=default --device=cpu --state_dict_path='models/big_matrix_default_explicit_negatives_with_weighted_dislike/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true --training_dataset='big_matrix' --weighted_dislike=true
```
## Methodology

We've explored multiple options at first:
- TODO Parler de BERT
- ALS
- We finally decided to implement the SASRec: Self-Attentive Sequential Recommendation Arhitecture.

We've found a Research paper [SASRec: Self-Attentive Sequential Recommendation](https://github.com/kang205/SASRec) and a base Pytorch implementation [SASRec-pytorch](https://github.com/pmixer/SASRec.pytorch) which we are using as a base implementation for our project.

### Model Architecture: SASRec (Self-Attentive Sequential Recommendation)

The implemented solution uses the SASRec architecture, which leverages the power of self-attention mechanisms for sequential recommendation. Key components include:

- **Self-Attention Layers**: Multi-head self-attention mechanism to capture complex item-item relationships
- **Point-Wise Feed-Forward Networks**: Two-layer neural networks applied after attention layers
- **Positional Embeddings**: To maintain sequential order information
- **Layer Normalization**: For training stability
- **Dropout**: For regularization and preventing overfitting

## Project

You can find multiple main parts in the project:
- The EDA notebook (exploratory data analysis)
- The prepare_data.py file which is used to prepare the data for the model (convert the raw data from KuaiRec 2.0 to the format required by SASRec)
- The main.py file which is the entry point of the project (see command to run at the beginning of the README), which is use in combination with model.py and utils.py

```sh
project
  ├── solution
  │   ├── EDA.ipynb
  │   ├── prepare_data.py
  │   ├── main.py
  │   └── model.py
```

## Experiments

We conducted an extensive hyperparameter search with the following parameters:

- Learning rates: [0.001, 0.0005]
- Maximum sequence lengths: [50, 100]
- Number of transformer blocks: [1, 2]
- Dropout rates: [0.2, 0.5]
- Hidden units: 50 (fixed)
- Number of attention heads: 1 (fixed)
- Training epochs: [2, 5]

### Evaluation Metrics

The model performance was evaluated using:
- NDCG@10 (Normalized Discounted Cumulative Gain)
- Precision@10
- Recall@10

## Results

### Best Performing Configuration

The best performing model achieved the following metrics:

- **Validation NDCG@10**: TODO
- **Test NDCG@10**: TODO
- **Test Precision@10**: TODO
- **Test Recall@10**: TODO

This was achieved with the following hyperparameters:
- Learning rate: TODO
- Maximum sequence length: TODO
- Number of blocks: TODO
- Dropout rate: TODO
- Training epochs: TODO

### Key Findings


## Conclusions


## Benchmark results

--- Top 10 Benchmark Results (Sorted by Test NDCG) ---
Rank | LR      | Maxlen | Blocks | Hidden | Epochs | Dropout | Val NDCG | Test NDCG | Val R@10 | Test R@10 | Duration (s)
-------------------------------------------------------------------------------------------------------------------------
1    | 0.00150 | 150    | 2      | 50     | 25     | 0.500   | 0.9874   | 0.9989    | 0.1974   | 0.0999    | 66.25       
2    | 0.00150 | 150    | 2      | 50     | 17     | 0.700   | 0.9879   | 0.9987    | 0.1975   | 0.0999    | 44.64       
3    | 0.00050 | 300    | 4      | 50     | 25     | 0.700   | 0.9872   | 0.9983    | 0.1973   | 0.0998    | 252.55      
4    | 0.00050 | 150    | 2      | 50     | 25     | 0.700   | 0.9871   | 0.9979    | 0.1976   | 0.0998    | 68.33       
5    | 0.00100 | 50     | 2      | 50     | 2      | 0.200   | 0.9868   | 0.9974    | 0.1975   | 0.0998    | 8.09        
6    | 0.00050 | 150    | 4      | 50     | 17     | 0.700   | 0.9879   | 0.9971    | 0.1974   | 0.0998    | 78.23       
7    | 0.00050 | 100    | 1      | 50     | 5      | 0.500   | 0.9875   | 0.9967    | 0.1975   | 0.0997    | 11.82       
8    | 0.00100 | 100    | 2      | 50     | 2      | 0.500   | 0.9869   | 0.9964    | 0.1976   | 0.0997    | 11.58       
9    | 0.00100 | 100    | 1      | 50     | 2      | 0.200   | 0.9872   | 0.9958    | 0.1975   | 0.0997    | 7.88        
10   | 0.00100 | 100    | 2      | 50     | 2      | 0.500   | 0.9870   | 0.9958    | 0.1973   | 0.0996    | 9.67        
-------------------------------------------------------------------------------------------------------------------------