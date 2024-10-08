import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import oddt.metrics as vsmetrics
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score, recall_score, precision_score, mean_absolute_error, average_precision_score
from scipy.stats import pearsonr, spearmanr
from utils.fingerprint import Fingerprint
from model.GeminiMol import PropDecoder
from functools import partial
from utils.chem import check_smiles_validity, gen_standardize_smiles

class QSAR:
    def __init__(
            self, 
            model_name, 
            encoder_list=[Fingerprint(['ECFP4'])], 
            standardize=True, 
            label_column='state', 
            smiles_column='smiles', 
        ):
        self.standardize = standardize
        self.model_name = model_name
        self.label_column = label_column
        self.smiles_column = smiles_column
        self.metric_functions = { # use these metrics to evaluate models
            'AUROC': lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
            'AUPRC': lambda y_true, y_pred: average_precision_score(y_true, y_pred),
            'AUROPRC': lambda y_true, y_pred: 0.5*roc_auc_score(y_true, y_pred)+0.5*average_precision_score(y_true, y_pred),
            'BEDROC': lambda y_true, y_pred: vsmetrics.bedroc(y_true, y_pred, alpha=160.9, pos_label=1),
            'EF1%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=1, pos_label=1, kind='fold'),
            'EF0.5%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.5, pos_label=1, kind='fold'),
            'EF0.1%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.1, pos_label=1, kind='fold'),
            'EF0.05%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.05, pos_label=1, kind='fold'),
            'EF0.01%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.01, pos_label=1, kind='fold'),
            'logAUC': lambda y_true, y_pred: vsmetrics.roc_log_auc(y_true, y_pred, pos_label=1, ascending_score=False, log_min=0.001, log_max=1.0),
            'MSE': lambda y_true, y_pred: -1*mean_squared_error(y_true, y_pred),
            'MAE': lambda y_true, y_pred: -1*mean_absolute_error(y_true, y_pred),
            'RMSE': lambda y_true, y_pred: -1*np.sqrt(mean_squared_error(y_true, y_pred)),
            'SPEARMANR': lambda y_true, y_pred: spearmanr(y_true, y_pred)[0],
            'PEARSONR': lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
            'ACC': lambda y_true, y_pred: accuracy_score(y_true, [round(num) for num in y_pred]),
            'specificity': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred], pos_label=0), 
            'precision': lambda y_true, y_pred: precision_score(y_true, [round(num) for num in y_pred]), 
            'recall': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred]), 
            'sensitivity': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred]), 
            'f1': lambda y_true, y_pred: f1_score(y_true, [round(num) for num in y_pred]), 
        }
        self.encoders = encoder_list
        if os.path.exists(f"{self.model_name}/predictor.pt"):
            with open(f"{self.model_name}/model_params.json", 'r', encoding='utf-8') as f:
                self.params = json.load(f)
            self.predictor = PropDecoder(
                feature_dim = self.params['feature_dim'],
                hidden_dim = self.params['hidden_dim'],
                expand_ratio = self.params['expand_ratio'],
                num_layers = self.params['num_layers'],
                dense_dropout = self.params['dense_dropout'],
                dropout_rate = self.params['dropout_rate'], 
                rectifier_activation = self.params['rectifier_activation'], 
                concentrate_activation = self.params['concentrate_activation'], 
                dense_activation = self.params['dense_activation'], 
                projection_activation = self.params['projection_activation'], 
                projection_transform = self.params['projection_transform'], 
                linear_projection = self.params['linear_projection'],
            )
            self.batch_size = self.params['batch_size']
            self.predictor.load_state_dict(torch.load(f'{self.model_name}/predictor.pt'))
        else:
            os.makedirs(self.model_name, exist_ok=True)
        self.all_metric = {
            'binary' : ['ACC', 'AUROC', 'AUPRC', 'recall', 'precision', 'f1'],
            'regression' : ['RMSE', 'MAE', 'MSE', 'PEARSONR', 'SPEARMANR'],
        }
        self.label_map = {
            'Active': 1, 
            'Inactive': 0, 
            'active': 1, 
            'inactive': 0, 
            'Yes': 1, 
            'No': 0, 
            'yes': 1, 
            'no': 0, 
            True: 1,
            False: 0,
            'True': 1, 
            'False': 0, 
            'true': 1, 
            'false': 0, 
            'Positive': 1, 
            'Negative': 0, 
            'positive': 1, 
            'negative': 0, 
            1: 1, 
            0: 0
        }

    def prepare(self, dataset):
        dataset = dataset.dropna(subset=[self.smiles_column])
        print(f"NOTE: read the dataset size ({len(dataset)}).")
        out_dataset = dataset.copy()
        if self.standardize == True:
            out_dataset.loc[:, self.smiles_column] = out_dataset[self.smiles_column].apply(lambda x: gen_standardize_smiles(x, kekule=False, random=False))
        else:
            out_dataset.loc[:, self.smiles_column] = out_dataset[self.smiles_column].apply(lambda x: check_smiles_validity(x))
        out_dataset = out_dataset[out_dataset[self.smiles_column]!='smiles_unvaild']
        print(f"NOTE: processed dataset size ({len(out_dataset)}).")
        return out_dataset.reset_index(drop=True)
    
    def parallel_encode(self, query_smiles_table, smiles_name = None):
        if smiles_name is None:
            smiles_name = self.smiles_column
        features_columns = []
        query_smiles_table = self.prepare(query_smiles_table)
        for single_encoder in self.encoders:
            features = single_encoder.extract_features(query_smiles_table, smiles_column=smiles_name)
            features_columns += list(features.columns)
            query_smiles_table = query_smiles_table.join(features, how='left')
        return query_smiles_table, features_columns

    def encode(self, smiles_list):
        features = torch.cat([
            single_encoder.encode(smiles_list).cuda()
            for single_encoder in self.encoders
            ], dim=1)
        return features

    def _predict(self, 
            df, 
            features_columns
        ):
        self.predictor.eval()
        with torch.no_grad():
            pred_values = []
            for i in range(0, len(df), self.batch_size):
                rows = df.iloc[i:i+self.batch_size]
                features = torch.tensor(rows[features_columns].values, dtype=torch.float32).cuda()
                pred = self.predictor(features).cuda()
                pred_values += list(pred.cpu().detach().numpy())
            res_df = pd.DataFrame(
                {f'pred_{self.label_column}': pred_values}, 
                columns=[f'pred_{self.label_column}']
            )
            res_df.columns = [f'pred_{self.label_column}']
            return res_df[f'pred_{self.label_column}'].to_list()

    def _evaluate(self, 
        df, 
        features_columns,
        label_name = 'label',
        metrics = ['SPEARMANR']
        ):
        y_pred = np.array(self._predict(df, features_columns))
        y_ture = np.array(df[label_name].to_list())
        results = {}
        for metric in metrics:
            results[metric] = self.metric_functions[metric](y_ture, y_pred)
        return results

    def trianing_models(self,
            training_set, 
            val_set, 
            epochs = 10,
            learning_rate = 1.0e-5,
            weight_decay = 0.001,
            patience = 50,
            optim_type = 'AdamW',
            params = {
                'task_type': 'binary',
                'hidden_dim': 1024,
                'expand_ratio': 3,
                'dropout_rate': 0.1,
                'num_layers': 3,
                'rectifier_activation': 'ELU',
                'concentrate_activation': 'ELU',
                'dense_activation': 'ELU',
                'projection_activation': 'ELU',
                'projection_transform': 'Sigmoid',
                'linear_projection': True,
                'batch_size': 128
            },
            mini_epoch = 200
        ):
        self.params = params
        # Load the models and optimizers
        models = {
            'AdamW': partial(
                torch.optim.AdamW, 
                lr = learning_rate, 
                betas = (0.9, 0.98),
                weight_decay = weight_decay
            ),
            'Adam': partial(
                torch.optim.Adam,
                lr = learning_rate, 
                betas = (0.9, 0.98),
                weight_decay = weight_decay
            ),
            'SGD': partial(
                torch.optim.SGD,
                lr=learning_rate, 
                momentum=0.8, 
                weight_decay=weight_decay
            ),
            'Adagrad': partial(
                torch.optim.Adagrad,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
            'Adadelta': partial(
                torch.optim.Adadelta,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
            'RMSprop': partial(
                torch.optim.RMSprop,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
            'Adamax': partial(
                torch.optim.Adamax,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
            'Rprop': partial(
                torch.optim.Rprop,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
        }
        if optim_type not in models:
            print(f"Invalid optimizer type {optim_type}. Defaulting to AdamW.")
            optim_type = 'AdamW'
        # setup task type
        if self.params['task_type'] == 'binary':
            training_set[self.label_column] = training_set[self.label_column].replace(self.label_map)
            val_set[self.label_column] = val_set[self.label_column].replace(self.label_map)
            label_set = list(set(training_set[self.label_column].to_list()))
            pos_num = len(training_set[training_set[self.label_column]==label_set[0]]) 
            neg_num = len(training_set[training_set[self.label_column]==label_set[1]])
            self.eval_metric = 'AUROC'
            self.loss_function = 'Focal' if pos_num/neg_num > 25 or neg_num/pos_num > 25 else 'BCE'
        else:
            self.eval_metric = 'SPEARMANR'
            self.loss_function = 'MSE'
        train_features, features_columns = self.parallel_encode(
            training_set, 
            smiles_name = self.smiles_column
        )
        val_features, features_columns = self.parallel_encode(
            val_set, 
            smiles_name = self.smiles_column
        )
        # model
        self.params['feature_dim'] = len(features_columns)
        self.predictor = PropDecoder(
            feature_dim = self.params['feature_dim'],
            hidden_dim = self.params['hidden_dim'],
            expand_ratio = self.params['expand_ratio'],
            num_layers = self.params['num_layers'],
            dense_dropout = self.params['dense_dropout'],
            dropout_rate = self.params['dropout_rate'], 
            rectifier_activation = self.params['rectifier_activation'], 
            concentrate_activation = self.params['concentrate_activation'], 
            dense_activation = self.params['dense_activation'], 
            projection_activation = self.params['projection_activation'], 
            projection_transform = self.params['projection_transform'], 
            linear_projection = self.params['linear_projection'],
        )
        self.batch_size, train_batch_size = self.params['batch_size'] * 4, self.params['batch_size']
        with open(f"{self.model_name}/model_params.json", 'w', encoding='utf-8') as f:
            json.dump(self.params, f, ensure_ascii=False, indent=4)
        # Set up the optimizer
        optimizer = models[optim_type](self.predictor.parameters())
        batch_id = 0
        best_score = -1.0
        val_features = val_features.reset_index(drop=True)
        patience_pool = 0 
        for _ in range(epochs):
            self.predictor.train()
            train_features = train_features.sample(frac=1).reset_index(drop=True)
            for i in range(0, len(train_features), train_batch_size):
                batch_id += 1
                rows = train_features.iloc[i:i+train_batch_size]
                if len(rows) <= 8:
                    continue
                features = torch.tensor(
                    rows[features_columns].values, 
                    dtype=torch.float32
                ).cuda()
                pred = self.predictor(features).cuda()
                label_tensor = torch.tensor(
                    rows[self.label_column].to_list(), dtype=torch.float32
                ).cuda()
                if self.loss_function == 'BCE':
                    loss = torch.mean(
                        nn.BCELoss(
                            reduction = 'none',
                        )(
                            pred, label_tensor
                        ) * torch.tensor([1 - rows[self.label_column].mean()]).cuda()
                    )
                elif self.loss_function == 'Focal':
                    (alpha, gamma) = (0.25, 5)
                    bce_loss = nn.BCELoss(
                            reduction = 'none',
                        )(
                            pred, 
                            label_tensor
                        )
                    focal_loss = alpha * (1 - torch.exp(-bce_loss)) ** gamma * bce_loss
                    loss = torch.mean(focal_loss)
                elif self.loss_function == 'MSE':
                    loss = nn.MSELoss()(
                        pred, label_tensor
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'Epoch: {_+1}, batch id: {batch_id}, loss: {loss.item()}')
                if batch_id % mini_epoch == 0:
                    val_res = self._evaluate( 
                            val_features, features_columns,
                            label_name = self.label_column,
                            metrics = [self.eval_metric]
                        )
                    self.predictor.train()
                    print(f"Epoch {_+1}, evaluate {self.eval_metric} on the validation set: {val_res[self.eval_metric]}")
                    if np.isnan(val_res[self.eval_metric]) and os.path.exists(f'{self.model_name}/predictor.pt'):
                        print("NOTE: The parameters don't converge, back to previous optimal model.")
                        self.predictor.load_state_dict(torch.load(f'{self.model_name}/predictor.pt'))
                        patience -= 2
                    elif best_score < val_res[self.eval_metric]:
                        patience += 1
                        patience_pool = 0
                        best_score = val_res[self.eval_metric]
                        torch.save(self.predictor.state_dict(), f'{self.model_name}/predictor.pt')
                    else:
                        patience_pool += 1
                        patience -= 1
                if patience <= 0:
                    print("NOTE: The parameters was converged, stop training!")
                    break
            val_res = self._evaluate( 
                    val_features, features_columns,
                    label_name = self.label_column,
                    metrics = [self.eval_metric]
                )
            self.predictor.train()
            print(f"Epoch {_+1}, evaluate {self.eval_metric} on the validation set: {val_res[self.eval_metric]}")
            if np.isnan(val_res[self.eval_metric]) and os.path.exists(f'{self.model_name}/predictor.pt'):
                print("NOTE: The parameters don't converge, back to previous optimal model.")
                self.predictor.load_state_dict(torch.load(f'{self.model_name}/predictor.pt'))
                patience -= 2
            elif best_score < val_res[self.eval_metric]:
                patience += 1
                patience_pool = 0
                best_score = val_res[self.eval_metric]
                torch.save(self.predictor.state_dict(), f'{self.model_name}/predictor.pt')
            else:
                patience_pool += 1
                patience -= 1
            if patience <= 0:
                break
        self.predictor.load_state_dict(torch.load(f'{self.model_name}/predictor.pt'))
        val_res = self._evaluate(
                val_features, features_columns, 
                label_name = self.label_column,
                metrics = self.all_metric[self.params['task_type']]
            )
        print(f"Training is over, evaluate on the validation set:")
        print(val_res)

    def predict(self, 
            df, 
            parallel = True,
            smiles_name = 'smiles',
        ):
        self.predictor.eval()
        ori_columns = df.columns
        if smiles_name in ori_columns:
            pass
        elif self.smiles_column in ori_columns:
            smiles_name = self.smiles_column
        else:
            raise RuntimeError(f"The test set need smiles columns {self.smiles_column}.")
        print("NOTE: making prediction ......")
        with torch.no_grad():
            pred_values = []
            if parallel == True:
                features_df, features_columns = self.parallel_encode(df, smiles_name = smiles_name)
                features_df = features_df.reset_index(drop=True)
                for i in range(0, len(features_df), self.batch_size):
                    rows = features_df.iloc[i:i+self.batch_size]
                    features = torch.tensor(rows[features_columns].values, dtype=torch.float32).cuda()
                    pred = self.predictor(features).cuda()
                    pred_values += list(pred.cpu().detach().numpy())
            else:
                for i in range(0, len(df), self.batch_size):
                    rows = df.iloc[i:i+self.batch_size]
                    smiles = rows[smiles_name].to_list()
                    features = self.encode(smiles)
                    pred = self.predictor(features).cuda()
                    pred_values += list(pred.cpu().detach().numpy())
            res_df = pd.DataFrame(
                {f'pred_{self.label_column}': pred_values}, 
                columns=[f'pred_{self.label_column}']
            )
            res_df.columns = [f'pred_{self.label_column}']
            return features_df[ori_columns].join(res_df, how='left')
            
    def evaluate(self, 
            df, 
            smiles_name = 'smiles', 
            label_name = 'label',
            metrics = ['SPEARMANR'],
            as_pandas = True
        ):
        assert label_name in df.columns, f"The test set need label columns {label_name}."
        assert smiles_name in df.columns, f"The test set need smiles columns {smiles_name}."
        df.dropna(
            subset=[smiles_name, label_name], 
            inplace=True
        )
        df.drop_duplicates(
            subset=[smiles_name], 
            keep='first', 
            inplace=True, 
            ignore_index=True
        )
        prediction = self.predict(df, smiles_name=smiles_name)
        if self.params['task_type'] == 'binary':
            prediction[label_name] = prediction[label_name].replace(self.label_map)
        print(prediction[[label_name, f'pred_{self.label_column}']])
        print("NOTE: evaluate the prediction ......")
        y_ture = np.array(prediction[label_name].to_list())
        y_pred = np.array(prediction[f'pred_{self.label_column}'].to_list())
        results = {}
        for metric in metrics:
            results[metric] = self.metric_functions[metric](y_ture, y_pred)
        if as_pandas:
            return pd.DataFrame(results, index=[0])
        else:
            return results

if __name__ == "__main__":
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # random_seed
    training_random_seed = 1207
    np.random.seed(training_random_seed)
    torch.manual_seed(training_random_seed)
    torch.cuda.manual_seed(training_random_seed) 
    torch.cuda.manual_seed_all(training_random_seed)
    # params
    data_path = sys.argv[1]
    target = os.path.basename(sys.argv[1])
    encoding_method = sys.argv[2]
    smiles_column = sys.argv[3]
    label_column = sys.argv[4]
    product_model_name = sys.argv[5]
    # read datasets
    train_data = pd.read_csv(f'{data_path}/{target}_scaffold_train.csv')
    print(
        f"{target} Training Set: Number=", 
        len(train_data), 
        )
    if len(list(set(train_data[label_column].to_list()))) == 2:
        task_type = 'binary'
        test_metrics = ['AUROC', 'BEDROC', 'ACC', 'f1', 'AUPRC']
    else:
        task_type = 'regression'
        test_metrics = ['SPEARMANR', 'RMSE', 'MAE', 'MSE', 'PEARSONR']
    val_data = pd.read_csv(f'{data_path}/{target}_scaffold_valid.csv')
    print(
        f"{target} Validation Set: Number=", 
        len(val_data), 
        )
    test_data = pd.read_csv(f'{data_path}/{target}_scaffold_test.csv')
    print(
        f"{target} Test Set: Number=", 
        len(test_data), 
        )
    ## read the encoder models
    fingerprint_list = []
    encoders = {}
    for model_name in encoding_method.split(":"):
        if os.path.exists(f'{model_name}/GeminiMol.pt'):
            method_list = [model_name]
            from model.GeminiMol import GeminiMol
            encoders[model_name] = GeminiMol(
                model_name,
                depth = 0, 
                custom_label = None, 
                extrnal_label_list = ['Cosine', 'Pearson', 'RMSE', 'Manhattan']
            )
        elif os.path.exists(f'{model_name}/backbone'):
            from model.CrossEncoder import CrossEncoder
            encoders[model_name] = CrossEncoder(
                model_name,
                candidate_labels = [
                    'LCMS2A1Q_MAX', 'LCMS2A1Q_MIN', 'MCMM1AM_MAX', 'MCMM1AM_MIN', 
                    'ShapeScore', 'ShapeOverlap', 'ShapeAggregation', 'CrossSim', 'CrossAggregation', 'CrossOverlap', 
                ]
            )
        elif model_name == "CombineFP":
            methods_list = ["ECFP4", "FCFP6", "AtomPairs", "TopologicalTorsion"]
            encoders[model_name] = Fingerprint(methods_list)
        else:
            methods_list = [model_name]
            fingerprint_list += [model_name]
    if len(fingerprint_list) > 0:
        encoders['Fingerprints'] = Fingerprint(fingerprint_list)
    encoders_list = list(encoders.values())
    # baseline model
    baseline = False
    if baseline:
        from AutoQSAR import AutoQSAR
        baseline_model = AutoQSAR(
                f"{product_model_name}_AutoQSAR", 
                encoder_list = encoders_list, 
                standardize = False, 
                label_column = label_column, 
                smiles_column = smiles_column, 
                task_type = task_type
            )
        baseline_model.trianing_models(
            train_data, 
            val_set = val_data, 
            stack = False,
            num_trials = 30
        )
        test_leaderboard = baseline_model.evaluate(test_data, metric_list=test_metrics)
        print(test_leaderboard)
        test_leaderboard.to_csv(f"{product_model_name}_baseline_results.csv", index=True)
    # setup QSAR models
    QSAR_model = QSAR(  
        f"{product_model_name}", 
        encoder_list = encoders_list, 
        standardize = False, 
        label_column = label_column, 
        smiles_column = smiles_column, 
    )
    # training QSAR models
    if not os.path.exists(f"{product_model_name}/predictor.pt"):
        QSAR_model.trianing_models(
            train_data,
            val_set=val_data,
            epochs = ( 300000 // len(train_data) ) + 1,
            learning_rate = 1.0e-4,
            params = {
                'task_type': task_type,
                'hidden_dim': 1024,
                'expand_ratio': 3,
                'dense_dropout': 0.0,
                'dropout_rate': 0.3 if task_type == 'binary' else 0.0,
                'num_layers': 3,
                'rectifier_activation': 'SiLU',
                'concentrate_activation': 'SiLU',
                'dense_activation': 'SiLU',
                'projection_activation': 'Softplus' if task_type == 'binary' else 'Identity',
                'projection_transform': 'Sigmoid' if train_data[label_column].max() <= 1.0 and train_data[label_column].min() >= 0.0 else 'Identity',
                'linear_projection': False,
                'batch_size': 32
            },
            patience = 120
        )
    val_res = QSAR_model.evaluate(
        val_data, 
        smiles_name = smiles_column, 
        label_name = label_column,
        metrics = test_metrics
        ).rename(index={0: 'Valid'})
    label_set = list(set(train_data[label_column].to_list()))
    pos_num = len(train_data[train_data[label_column]==label_set[0]]) 
    neg_num = len(train_data[train_data[label_column]==label_set[1]])
    if len(label_set) == 2:
        task_type = 'binary'
        if pos_num/neg_num > 100.0 or neg_num/pos_num > 100.0:
            core_metrics = 'BEDROC'
        elif pos_num/neg_num > 3.0 or neg_num/pos_num > 3.0:
            core_metrics = 'AUPRC'
        else:
            core_metrics = 'AUROC'
    else:
        core_metrics = 'SPEARMANR'
    test_res = QSAR_model.evaluate(
        test_data, 
        smiles_name = smiles_column, 
        label_name = label_column,
        metrics = test_metrics,
        as_pandas= False
        )
    core_score = test_res[core_metrics]
    total_res = pd.concat([val_res, pd.DataFrame(test_res, index=['Test'])])
    total_res.to_csv(f"{product_model_name}/results.csv", index=True, header=True, sep=',')
    print('======== Job Report ========')
    print(f'{target} Model performance: ')
    print(total_res)
    if baseline:
        print('Baseline performance on the test set: ')
        print(test_leaderboard)
    print(f"NOTE: The {core_metrics} on the {target}: {core_score}")
    print('============================')



