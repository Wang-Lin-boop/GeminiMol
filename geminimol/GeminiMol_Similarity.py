import sys
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.GeminiMol import GeminiMol
from utils.fingerprint import Fingerprint

def find_closest_factors(x):
    factors = []
    for i in range(1, int(math.sqrt(x)) + 1):
        if x % i == 0:
            factors.append([i, x // i])
    min_diff = float('inf')
    result = (None, None)
    for i in range(len(factors)):
        n = factors[i][0]
        m = factors[i][1]
        diff = abs(n - m)
        if diff < min_diff:
            min_diff = diff
            result = (n, m)
    return result

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # defalut params
    training_random_seed = 1207
    np.random.seed(training_random_seed)   
    model_name = sys.argv[1]
    simles_1 = str(sys.argv[2])
    simles_2 = str(sys.argv[3])
    jobname = str(sys.argv[4])
    predictor_geminimol = GeminiMol(model_name = model_name)
    predictor_fingerprint = Fingerprint()
    print(f"NOTE: Loading the {model_name} ...")
    predictor_geminimol.similarity_metrics_list = [
        'Cosine', 'Manhattan', 'Euclidean', 'Pearson',
        'ShapeScore', 'ShapeOverlap', 'ShapeDistance','ShapeAggregation'
    ]
    input_sents = [simles_1, simles_2, simles_2, simles_1]
    features = predictor_geminimol.encode(input_sents)
    _, feature_size = features.size()
    print(f'NOTE: feature size: {feature_size}')
    n, m = find_closest_factors(feature_size)
    pred_values = {
        **{
            'smiles1': [simles_1, simles_2],
            'smiles2': [simles_2, simles_1],
        }, 
        **{ key:[] for key in predictor_geminimol.similarity_metrics_list}
    }
    for label_name in predictor_geminimol.similarity_metrics_list:
        pred = predictor_geminimol.decode(features, label_name, 2)
        pred_values[label_name] += list(pred.cpu().detach().numpy())
    data_table = pd.DataFrame(pred_values)
    data_table['MACCS_Tanimoto'] = data_table.apply(lambda x: predictor_fingerprint.similarity(x['smiles1'], x['smiles2'], 'MACCS', 'Tanimoto'), axis=1)
    data_table['ECFP4_Tversky'] = data_table.apply(lambda x: predictor_fingerprint.similarity(x['smiles1'], x['smiles2'], 'ECFP4', 'Tversky'), axis=1)
    data_table['TopologicalTorsion_Tversky'] = data_table.apply(lambda x: predictor_fingerprint.similarity(x['smiles1'], x['smiles2'], 'TopologicalTorsion', 'Tversky'), axis=1)
    data_table['AtomPairs_Tanimoto'] = data_table.apply(lambda x: predictor_fingerprint.similarity(x['smiles1'], x['smiles2'], 'AtomPairs', 'Tanimoto'), axis=1)
    data_table.to_csv(f'{jobname}_pred.csv', index=False)
    print(f'NOTE: feature re-shape: {n}x{m}')
    feature_1 = features[0].view(n, m).cpu().detach().numpy()
    feature_2 = features[1].view(n, m).cpu().detach().numpy()
    features_diff = feature_1 - feature_2
    plt.imshow(features_diff, cmap='hot', vmin=-0.3, vmax=0.3)
    plt.colorbar().remove()
    plt.tight_layout()
    plt.savefig(f'{jobname}_featdiff.png')
    plt.close()










