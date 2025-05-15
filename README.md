


Prepare Data 

```
# this step is a pre-processing step to build the data for training, converting the raw format from KuaiRec 2.0 to the format required by SASRec
python prepare_data.py 
```


To launch the train
```shell
python main.py --dataset='small_matrix' --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cpu --num_epochs=20
```

To compute the inference only
```shell
python main.py --dataset='small_matrix' --train_dir=default --device=cpu --state_dict_path='models/small_matrix_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true
```

To generate the recommendations
```shell
python main.py --dataset='small_matrix' --train_dir=default --device=cpu --state_dict_path='models/small_matrix_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --generate_recommendations=true
```

To compute the inference only with a test dataset different than the training dataset
```shell
python main.py --dataset='small_matrix_no_remapping' --train_dir=default --device=cpu --state_dict_path='models/big_matrix_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true --training_dataset='big_matrix'
```