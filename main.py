from recbole_ccml.MetaUtils import metaQuickStart

param_config = {
    'metrics': ['mae', 'mse', 'rmse', 'ndcg'],
    'metric_decimal_place': 4,
    'topk': [1, 3, 5, 7],
    'valid_metric': 'mse',
    'local_update_count': 3,
    'user_inter_num_interval': [13, 100],
    'epochs': 500,
    'eval_args': {'group_by': 'task', 'order': 'RO', 'split': {'RS': [0.7, 0.1, 0.2]}, 'mode': 'labeled'},
    'meta_args': {'support_num': 10, 'query_num': 'none'},
    'train_batch_size': 32,

    'use_avg_grad': False,  # False
    'use_film': True,  # True
    'use_mlp_for_weight': False,  # False

    'vae_encoder_hidden_size': [128, 64, 20],

    'similar_task_weight': 0.1,
}
datasetName = 'ml-1m'
modelNames = ['CCML']

for modelName in modelNames:
    config, test_result = metaQuickStart(modelName, datasetName, param_config)
