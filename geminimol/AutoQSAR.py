import os
import sys
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from utils.fingerprint import Fingerprint
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score, recall_score, precision_score, mean_absolute_error, average_precision_score
import oddt.metrics as vsmetrics
from scipy.stats import pearsonr, spearmanr
from utils.chem import check_smiles_validity, gen_standardize_smiles

class AutoQSAR():
    def __init__(
            self, 
            model_name, 
            encoder_list=[Fingerprint(['ECFP4'])], 
            standardize=True, 
            label_column='state', 
            smiles_column='smiles', 
            task_type="binary"
        ):
        self.standardize = standardize
        self.QSAR_model_name = model_name
        self.label_column = label_column
        self.smiles_column = smiles_column
        self.task_type = task_type
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
        self.QSAR_encoder = encoder_list
        if os.path.exists(f"{self.QSAR_model_name}/predictor.pkl"):
            self.QSAR_Model = TabularPredictor.load(path=self.QSAR_model_name, verbosity=4)
        self.recommanded_metric = { # use this metrics in autogluon
            'binary' : 'roc_auc',
            'regression' : 'root_mean_squared_error', # for sorting the models, the root_mean_squared_error is opposite number of itself
        }
        self.all_metric = { # use this metrics in autogluon
            'binary' : ['accuracy', 'roc_auc', 'recall', 'precision', 'f1', 'log_loss'],
            'regression' : ['root_mean_squared_error', 'mean_absolute_error', 'mean_squared_error', 'pearsonr', 'spearmanr','r2'], # for sorting the models, the root_mean_squared_error, mean_absolute_error, and mean_squared_error is opposite number of themselves
        }
        self.autogluon_models = {
            'NN_TORCH': [
                {}
            ],
            'GBM': [
                {}, 
                {
                    'extra_trees': True, 
                    'ag_args': {'name_suffix': 'XT'}
                    }, 
                'GBMLarge',
                ], 
            'CAT': [
                {}, 
            ],
            'XGB': [
                {}, 
            ],
            'RF': [
                {
                    'criterion': 'gini', 
                    'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}
                    }, 
                {
                    'criterion': 'entropy', 
                    'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}
                    }, 
                {
                    'criterion': 'squared_error', 
                    'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}
                    },
                ], 
            'XT': [
                {
                    'criterion': 'gini', 
                    'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}
                    }, 
                {
                    'criterion': 'entropy', 
                    'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}
                    }, 
                {
                    'criterion': 'squared_error', 
                    'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}
                    },
                ],
            'KNN': [
                {
                    'weights': 'uniform', 
                    'ag_args': {'name_suffix': 'Unif'}
                    }, 
                {
                    'weights': 'distance', 
                    'ag_args': {'name_suffix': 'Dist'}
                    },
            ],
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

    def encoder(self, query_smiles_table):
        features_columns = []
        query_smiles_table = self.prepare(query_smiles_table)
        for single_encoder in self.QSAR_encoder:
            features = single_encoder.extract_features(query_smiles_table, smiles_column=self.smiles_column)
            features_columns += list(features.columns)
            query_smiles_table = query_smiles_table.join(features, how='left')
        return query_smiles_table, features_columns
    
    def trianing_models(self, 
            training_set, 
            val_set = None, 
            eval_metric = None, 
            stack = False, 
            num_trials = 1,
            model_list = ['GBM', 'NN_TORCH', 'CAT', 'XGB']
        ):
        if stack:
            presets = {
                'auto_stack': True, 
                'refit_full': 'all',
                }
            training_set = pd.concat([training_set, val_set], ignore_index=True)
            val_set = None
        else:
            presets = {
                'auto_stack': False, 
                'refit_full': 'all',
                }
        training_set, features_columns = self.encoder(training_set)
        print(f"NOTE: The {len(features_columns)} features were extracted from training set.")
        if eval_metric is None or eval_metric == 'None':
            eval_metric = self.recommanded_metric[self.task_type]
        self.QSAR_Model = TabularPredictor(label=self.label_column, path=self.QSAR_model_name, eval_metric=eval_metric, problem_type=self.task_type)
        if val_set is None:
            tuning_data = None
        else:
            val_set, _ = self.encoder(val_set)
            tuning_data = val_set[features_columns+[self.label_column]]
        if num_trials > 1:
            self.QSAR_Model.fit(
                train_data = shuffle(training_set[features_columns+[self.label_column]]), 
                tuning_data = tuning_data,
                num_cpus = 4,
                num_gpus = 1,
                verbosity = 4, 
                presets = presets,
                hyperparameters = self.autogluon_models,
                included_model_types = model_list,
                feature_prune_kwargs = {},
                hyperparameter_tune_kwargs = {
                    'num_trials': num_trials,
                    'scheduler' : 'local',
                    'searcher': 'auto',
                } 
            )
        else:
            self.QSAR_Model.fit(
                train_data = shuffle(training_set[features_columns+[self.label_column]]), 
                tuning_data = tuning_data,
                num_cpus = 4,
                num_gpus = 1,
                verbosity = 4, 
                presets = presets,
                hyperparameters = self.autogluon_models,
                included_model_types = model_list,
                feature_prune_kwargs = {},
            )
        self.best_model_name = self.QSAR_Model.get_model_best()
        print(f"NOTE: The best model of {self.QSAR_model_name} is {self.best_model_name}.")
    
    def predict(self, test_set, model=None):
        assert self.smiles_column in test_set.columns, f"The test set need smiles columns {self.smiles_column}."
        if model is None:
            model = self.best_model_name
        orig_columns = test_set.columns
        test_set, _ = self.encoder(test_set)
        print("NOTE: making prediction ......")
        if self.task_type == 'binary':
            predicted_score = self.QSAR_Model.predict_proba(test_set, model=model)[[1]]
        else:
            predicted_score = self.QSAR_Model.predict(test_set, model=model)
            predicted_score = pd.DataFrame(predicted_score, columns=[self.label_column])
        predicted_score.columns = ['pred']
        return test_set[orig_columns].join(predicted_score, how='left')
        
    def test(self, test_set):
        assert self.label_column in test_set.columns, f"The test set need label columns {self.label_column}."
        assert self.smiles_column in test_set.columns, f"The test set need smiles columns {self.smiles_column}."
        test_set, _ = self.encoder(test_set)
        leaderboard_table = self.QSAR_Model.leaderboard(data=test_set, extra_info=True, silent=True, extra_metrics=self.all_metric[self.task_type])
        leaderboard_table.dropna(
            axis = 0, 
            how = 'all', 
            subset = [self.recommanded_metric[self.task_type]],
        )
        test_results = pd.DataFrame(leaderboard_table[['model']+self.all_metric[self.task_type]])
        test_results.sort_values(
            self.recommanded_metric[self.task_type], 
            ascending=False, 
            inplace=True,
            ignore_index = True
            ) # in autogluon leaderboard, the root_mean_squared_error is -1*rmse
        test_results['ID'] = self.QSAR_model_name
        test_results.to_csv(str(self.QSAR_model_name+"_results.csv"), index=False, header=True, sep=',')
        self.model_name_list = test_results['model'].to_list()
        self.best_model_name = self.model_name_list[0]
        return test_results

    def select_model(self, test_set, metric, as_pandas=False):
        assert self.label_column in test_set.columns, f"The test set need label columns {self.label_column}."
        assert self.smiles_column in test_set.columns, f"The test set need smiles columns {self.smiles_column}."
        test_set, _ = self.encoder(test_set)
        model_name_list = self.QSAR_Model.get_model_names()
        statistic_results = {}
        y_true = np.array(test_set[self.label_column].to_list())
        for intrnal_model_name in model_name_list:
            if self.task_type == 'binary':
                predicted_score = self.QSAR_Model.predict_proba(test_set, model=intrnal_model_name)
                assert 1 in predicted_score.columns, "Error: the pos label (1) not found."
                predicted_score = predicted_score[[1]]
            else:
                predicted_score = self.QSAR_Model.predict(test_set, model=intrnal_model_name)
                predicted_score = pd.DataFrame(predicted_score, columns=[self.label_column])
            predicted_score.columns = ['pred']
            y_pred = np.array(predicted_score['pred'].to_list())
            statistic_results[intrnal_model_name] = self.metric_functions[metric](y_true, y_pred)
        self.best_model_name = max(statistic_results, key=statistic_results.get)
        if as_pandas:
            return pd.DataFrame.from_dict(statistic_results, orient='index', columns=[metric_function])
        else:
            return statistic_results

    def evaluate(
            self, 
            test_set, 
            model_list=None, 
            metric_list=['AUROC', 'AUPRC'], 
            as_pandas=True
        ):
        assert self.label_column in test_set.columns, f"The test set need label columns {self.label_column}."
        assert self.smiles_column in test_set.columns, f"The test set need smiles columns {self.smiles_column}."
        if model_list is None:
            model_list = [self.best_model_name]
        elif model_list == 'all':
            model_list = self.QSAR_Model.get_model_names()
        test_set.drop_duplicates(
            subset = [self.smiles_column], 
            keep = 'first', 
            inplace = True,
            ignore_index = True
        )
        test_set, _ = self.encoder(test_set)
        y_true = np.array(test_set[self.label_column].to_list())
        statistic_results = {'model_name': []}
        for metric in metric_list:
            statistic_results[metric] = []
        for model_name in model_list:
            statistic_results['model_name'] += [model_name]
            if self.task_type == 'binary':
                predicted_score = self.QSAR_Model.predict_proba(test_set, model=model_name)
                assert 1 in predicted_score.columns, "Error: the pos label (1) not found."
                predicted_score = predicted_score[[1]]
            else:
                predicted_score = self.QSAR_Model.predict(test_set, model=model_name)
                predicted_score = pd.DataFrame(predicted_score, columns=[self.label_column])
            predicted_score.columns = ['pred']
            y_pred = np.array(predicted_score['pred'].to_list())
            for metric in metric_list:
                statistic_results[metric] += [self.metric_functions[metric](y_true, y_pred)]
        if as_pandas:
            return pd.DataFrame(statistic_results, index=statistic_results['model_name'], columns=metric_list)
        else:
            return statistic_results

if __name__ == "__main__":
    target = sys.argv[1]
    method = sys.argv[2]
    smiles_column = sys.argv[3]
    label_column = sys.argv[4]
    eval_metric = sys.argv[5]
    metric_function = sys.argv[6]
    model_name = sys.argv[7]
    if len(sys.argv) > 7:
        extrnal_data = pd.read_csv(sys.argv[8])
    else:
        extrnal_data = None
    train_data = pd.read_csv(f'{target}/{target}_scaffold_train.csv')
    print(f"{target} Training Set: Number=", len(train_data), f", {len(train_data[train_data[label_column]==1])} rows is 1(pos).")
    if len(list(set(train_data[label_column].to_list()))) == 2:
        task_type = 'binary'
        test_metrics = ['AUROC', 'ACC', 'BEDROC', 'specificity', 'precision', 'recall', 'sensitivity', 'f1', 'AUPRC']
    elif 3 <= len(list(set(train_data[label_column].to_list()))) <= 10:
        task_type = 'multiclass'
        test_metrics = ['ACC']
    else:
        task_type = 'regression'
        test_metrics = ['RMSE', 'MAE', 'MSE', 'PEARSONR', 'SPEARMANR']
    val_data = pd.read_csv(f'{target}/{target}_scaffold_valid.csv')
    print(f"{target} Validation Set: Number=", len(val_data), f", {len(val_data[val_data[label_column]==1])} rows is 1(pos).")
    test_data = pd.read_csv(f'{target}/{target}_scaffold_test.csv')
    print(f"{target} Test Set: Number=", len(test_data), f", {len(test_data[test_data[label_column]==1])} rows is 1(pos).")
    if method == "CombineFP":
        methods_list = ["ECFP4", "FCFP6", "AtomPairs", "TopologicalTorsion"]
        encoders = [Fingerprint(methods_list)]
    else:
        methods_list = [method]
        encoders = [Fingerprint([methods_list])]
    target = str(target.split('/')[-1])
    if os.path.exists(f"{model_name}"):
        QSAR_model = AutoQSAR(
            f"{model_name}", 
            encoder_list = encoders, 
            standardize = True, 
            label_column = label_column, 
            smiles_column = smiles_column, 
            task_type = task_type
        )
    else:
        QSAR_model = AutoQSAR(
            f"{model_name}", 
            encoder_list = encoders, 
            standardize = True, 
            label_column = label_column, 
            smiles_column = smiles_column, 
            task_type = task_type
        )
        QSAR_model.trianing_models(
            train_data, 
            val_set = val_data, 
            eval_metric = eval_metric, 
            stack = False,
            num_trials = 1
        )
    leaderboard = QSAR_model.select_model(val_data, metric_function)
    print(f"NOTE: the best model on val set under the {metric_function} metric is {QSAR_model.best_model_name}, reached {leaderboard[QSAR_model.best_model_name]}.")
    print(f'NOTE: testing all models on the validation set ...')
    val_res = QSAR_model.evaluate(val_data, model_list=QSAR_model.QSAR_Model.get_model_names(), metric_list=test_metrics)
    print(val_res)
    val_res.to_csv(f"{model_name}/val_results.csv", index=True, header=True, sep=',')
    print(f'NOTE: testing all models on the test set ...')
    test_res = QSAR_model.evaluate(test_data, model_list=QSAR_model.QSAR_Model.get_model_names(), metric_list=test_metrics)
    print(test_res)
    test_res.to_csv(f"{model_name}/test_results.csv", index=True, header=True, sep=',')
    test_res.sort_values(
        metric_function, 
        ascending=False, 
        inplace=True,
        ignore_index = True
        )
    bestmodel_on_val = QSAR_model.best_model_name
    bestmodel_on_test = test_res[metric_function].idxmax()
    if bestmodel_on_val != bestmodel_on_test:
        print(f"Warning: the best model on test set is {bestmodel_on_test}, not {bestmodel_on_val}.")
    else:
        print(f"NOTE: the best model on test set is {bestmodel_on_test}, same as best model on val set.")
    if extrnal_data is not None:
        predicted_res = QSAR_model.predict(extrnal_data)
        predicted_res.to_csv(f"{model_name}_{QSAR_model.best_model_name}_prediction.csv", index=False, header=True, sep=',')



