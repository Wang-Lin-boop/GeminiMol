import os
import sys
import json
import torch
import pandas as pd
from sklearn.utils import shuffle
from model.CrossEncoder import CrossSimilarity, CrossEncoder
from benchmark import Benchmark

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # params
    training_random_seed = 1207
    data_path = sys.argv[1]
    bb_path = sys.argv[2]
    epochs = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    batch_size = int(sys.argv[5])
    model_name = sys.argv[6]
    if len(sys.argv) > 8:
        label_list = sys.argv[8].split(':')
    else:
        label_list = [
            'ShapeScore', 'ShapeAggregation', 'ShapeOverlap', 
            'CrossSim', 'CrossAggregation', 'CrossOverlap'
            ]
        # ['ShapeScore', 'ShapeOverlap', 'ShapeDistance', 'ShapeAggregation', 'LCMS2A1Q_MAX',  'MCMM1AM_MAX', "LCMS2A1Q_MIN", "MCMM1AM_MIN", 'CrossSim', 'CrossDist', 'CrossOverlap', 'CrossAggregation', "MCS"]
    if epochs > 0:
        # initial a DeepShape BinarySimilarity model  
        predictor_deepshape = CrossSimilarity(model_name=model_name, bb_path=bb_path, label = label_list, feature_list = ['smiles1', 'smiles2'])
        print(f"NOTE: Training {model_name} ...")
        # read data and build dataset
        data_set = pd.read_csv(f'{data_path}/test.csv') 
        data_set = data_set[label_list+['smiles1', 'smiles2', 'assign']]
        val_set = shuffle(data_set[data_set['assign']=="val"], random_state=training_random_seed)
        cross_set = shuffle(data_set[data_set['assign']=="cross"], random_state=training_random_seed)
        test_set = shuffle(data_set[data_set['assign']=="test"], random_state=training_random_seed)
        calibration_set = pd.read_csv(f'{data_path}/calibration.csv') # ShapeScore >= 0.75 and MCS < 0.4 in training set
        adjacent_set = pd.read_csv(f'{data_path}/indep_adjacent.csv') # ShapeScore > 0.6 and MCS < 0.4 in val and test set
        # data process
        print('NOTE: Validation Set: Number=', len(val_set))
        print('NOTE: Test Set: Number=', len(test_set))
        print('NOTE: Cross Set: Number=', len(cross_set))
        del data_set
        train_set = shuffle(pd.read_csv(f'{data_path}/enhanced_training.csv'), random_state=training_random_seed)
        print('NOTE: Training Set: Number=', len(train_set))
        train_set = train_set[label_list+['smiles1', 'smiles2', 'assign']]
        # training 
        for label in label_list:
            print(f'NOTE: Train the {label} model ......')
            predictor_deepshape.fit(
                    label, 
                    train_set,  
                    val_df = pd.concat([val_set, calibration_set], ignore_index=True), 
                    epochs = epochs, 
                    learning_rate = learning_rate,
                    batch_size = batch_size, 
                    num_gpus = 4
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
    Benchmark_Protocol = Benchmark(CrossEncoder(model_name), model_name=model_name)
    # benchmarking on all Datasets
    benchmark_index_file = sys.argv[7]
    benchmark_file_basepath = os.path.dirname(benchmark_index_file)
    with open(benchmark_index_file, 'r', encoding='utf-8') as f:
        benchmark_index_dict = json.load(f)
    for benchmark_task in ['DUDE', 'LIT-PCBA', 'TIBD', 'ADMET-C', 'ADMET-R', 'QSAR', 'CELLS']:
        Benchmark_Protocol(benchmark_task, f"{benchmark_file_basepath}/{benchmark_index_dict[benchmark_task]}", standardize=False)
    

    