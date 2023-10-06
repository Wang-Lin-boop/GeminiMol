import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from model.GeminiMol import BinarySimilarity, GeminiMol
from benchmark import Benchmark

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # defalut params
    training_random_seed = 1207
    np.random.seed(training_random_seed)   
    # read data and build dataset
    data_path = sys.argv[1]
    # set training params
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    encoder_activation = sys.argv[4].split(':')[0]
    decoder_activation = sys.argv[4].split(':')[1]
    readout_type = sys.argv[5].split(':')[0]
    num_features = int(sys.argv[5].split(':')[1])
    num_layers = int(sys.argv[5].split(':')[2])
    encoding_features = int(sys.argv[5].split(':')[3])
    if readout_type in ['Mixed', 'MixedMLP']:
        integrate_layer_type = sys.argv[5].split(':')[4]
        integrate_layer_num = int(sys.argv[5].split(':')[5])
    else:
        integrate_layer_type = 'None'
        integrate_layer_num = 0
    decoder_expand_ratio = int(sys.argv[5].split(':')[6])
    decoder_dropout = float(sys.argv[5].split(':')[7])
    label_dict = {} # ShapeScore:0.2,ShapeAggregation:0.2,ShapeOverlap:0.1,CrossSim:0.2,CrossAggregation:0.1,MCS:0.2
    for label in str(sys.argv[6]).split(','):
        label_dict[label.split(':')[0]] = float(label.split(':')[1])
    model_name = sys.argv[7]
    # read data and build dataset
    data_set = pd.read_csv(f'{data_path}/test.csv') 
    val_set = shuffle(data_set[data_set['assign']=="val"], random_state=training_random_seed)
    cross_set = shuffle(data_set[data_set['assign']=="cross"], random_state=training_random_seed)
    test_set = shuffle(data_set[data_set['assign']=="test"], random_state=training_random_seed)
    # data process
    print('Validation Set: Number=', len(val_set))
    print('Test Set: Number=', len(test_set))
    print('Cross Set: Number=', len(cross_set))
    del data_set 
    # training
    # initial a GraphShape BinarySimilarity model  
    params = {
        "feature_list": ['smiles1','smiles2'],
        "batch_size": batch_size,
        "encoder_activation": encoder_activation,
        "decoder_activation": decoder_activation,
        "readout_type": readout_type,
        "num_features": num_features,
        "num_layers": num_layers,
        "integrate_layer_type": integrate_layer_type,
        "integrate_layer_num": integrate_layer_num,
        "encoding_features": encoding_features,
        "decoder_expand_ratio" : decoder_expand_ratio,
        "decoder_dropout_rate": decoder_dropout,
        "label_dict": label_dict,
    }
    predictor_deepshape = BinarySimilarity(model_name = model_name, **params)
    with open(f"{model_name}/model_params.json", 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    print(f"NOTE: Training {model_name} ...")
    print("NOTE: the params shown as follow:")
    print(params)
    if epochs > 0:
        train_set = shuffle(pd.read_csv(f'{data_path}/training_graph.csv'))
        print('Training Set: Number=', len(train_set))
        # training 
        predictor_deepshape.fit(
            train_df = train_set,  
            val_df = val_set, 
            epochs = epochs, 
            learning_rate = 1.0e-4, # 3.0e-5
            optim_type='AdamW',
            num_warmup_steps=30000,
            T_max=10000,
        )  
        val_score = predictor_deepshape.evaluate(val_set)
        print('Model performance on the validation set: ', model_name)
        print(val_score)
        # test best model
        test_score = predictor_deepshape.evaluate(test_set)
        print('======== Job Report ========')
        print('Model performance on the testing set: ', model_name)
        print(test_score)
        test_score.to_csv(str(model_name+"/"+model_name+"_testset_results.csv"), index=True, header=True, sep=',')
        cross_score = predictor_deepshape.evaluate(cross_set)
        print('Model performance on the crossing set: ', model_name)
        print(cross_score)
        cross_score.to_csv(str(model_name+"/"+model_name+"_crossset_results.csv"), index=True, header=True, sep=',')
    del predictor_deepshape
    Benchmark_Protocol = Benchmark(predictor=GeminiMol(model_name), model_name=model_name)
    # benchmarking on all datasets
    benchmark_index_file = sys.argv[8]
    with open(benchmark_index_file, 'r', encoding='utf-8') as f:
        benchmark_index_dict = json.load(f)
    Benchmark_Protocol('DUDE', benchmark_index_dict['DUDE'], standardize=False)
    Benchmark_Protocol('LIT-PCBA', benchmark_index_dict['LIT-PCBA'], standardize=False)
    Benchmark_Protocol('ADMET-C', benchmark_index_dict['ADMET-C'], standardize=False)
    Benchmark_Protocol('ADMET-R', benchmark_index_dict['ADMET-R'], standardize=False)



    