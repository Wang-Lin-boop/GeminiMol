import os
import sys
import json
import torch
import pandas as pd
from sklearn.utils import shuffle
from model.GeminiMol import BinarySimilarity, GeminiMol
from benchmark import Benchmark

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # read data and build dataset
    data_path = sys.argv[1]
    # set training params
    training_random_seed = 1207
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    gnn_type = sys.argv[4]
    readout_type = sys.argv[5].split(':')[0]
    num_features = int(sys.argv[5].split(':')[1])
    num_layers = int(sys.argv[5].split(':')[2])
    encoding_features = int(sys.argv[5].split(':')[3])
    if readout_type in ['Mixed', 'MixedMLP', 'MixedBN', 'AttentiveMLP', 'WeightedMLP', 'MMLP']:
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
    patience = int(sys.argv[8])
    # read data and build dataset
    data_set = pd.read_csv(f'{data_path}/test.csv') 
    val_set = shuffle(data_set[data_set['assign']=="val"], random_state=training_random_seed)
    cross_set = shuffle(data_set[data_set['assign']=="cross"], random_state=training_random_seed)
    test_set = shuffle(data_set[data_set['assign']=="test"], random_state=training_random_seed)
    calibration_set = pd.read_csv(f'{data_path}/calibration.csv') # ShapeScore >= 0.75 and MCS < 0.4 in training set
    adjacent_set = pd.read_csv(f'{data_path}/indep_adjacent.csv') # ShapeScore > 0.6 and MCS < 0.4 in val and test set
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
        "encoder_activation": 'LeakyReLU',
        "decoder_activation": 'LeakyReLU',
        "gnn_type": gnn_type,
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
        train_set = shuffle(pd.read_csv(f'{data_path}/training.csv'), random_state=training_random_seed)
        print('Training Set: Number=', len(train_set))
        # training 
        predictor_deepshape.fit(
            train_set = train_set,  
            val_set = val_set, 
            calibration_set = calibration_set,
            epochs = epochs, 
            learning_rate = 1.0e-4, # 3.0e-5
            optim_type='AdamW',
            num_warmup_steps=30000,
            T_max=10000,
            weight_decay=0.02,
            patience=patience
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
        adjacent_score = predictor_deepshape.evaluate(adjacent_set)
        print('Model performance on the css_adjacent set: ', model_name)
        print(adjacent_score)
        adjacent_score.to_csv(str(model_name+"/"+model_name+"_css_adjacent_results.csv"), index=True, header=True, sep=',')
    del predictor_deepshape
    Benchmark_Protocol = Benchmark(
        predictor = GeminiMol(
            model_name,
            extrnal_label_list=['RMSE', 'Cosine', 'Manhattan', 'Minkowski', 'Euclidean', 'KLDiv', 'Pearson']
            ), 
        model_name = model_name
    )
    # benchmarking on all datasets
    benchmark_index_file = sys.argv[9]
    benchmark_file_basepath = os.path.dirname(benchmark_index_file)
    with open(benchmark_index_file, 'r', encoding='utf-8') as f:
        benchmark_index_dict = json.load(f)
    for benchmark_task in ['DUDE', 'LIT-PCBA', 'TIBD', 'ADMET-C', 'ADMET-R', 'QSAR', 'CELLS']:
        Benchmark_Protocol(benchmark_task, f"{benchmark_file_basepath}/{benchmark_index_dict[benchmark_task]}", standardize=False)




    