



To launch the train
```shell
python train.py --dataset='train' --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cpu --num_epochs=10
```

To compute the inference only
```shell
python train.py --dataset='train' --train_dir=default --device=cpu --state_dict_path='train_default/SASRec.epoch=80.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true
```

To generate the recommendations
```shell
python train.py --dataset='train' --train_dir=default --device=cpu --state_dict_path='train_default/SASRec.epoch=80.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --generate_recommendations=true
```