import os


def build_experiment_paths(args):
    """
    Build artifact roots for the current experiment.

    Empty experiment_dir keeps the legacy layout:
      ./saved_models, ./saved_results, ./logs
    Non-empty experiment_dir nests those roots underneath that directory.
    """
    experiment_dir = getattr(args, 'experiment_dir', '') or '.'
    return {
        'experiment_dir': '' if experiment_dir == '.' else experiment_dir,
        'saved_models_root': os.path.join(experiment_dir, 'saved_models'),
        'saved_results_root': os.path.join(experiment_dir, 'saved_results'),
        'logs_root': os.path.join(experiment_dir, 'logs'),
        'model_dataset_dir': os.path.join(experiment_dir, 'saved_models', args.model_name, args.dataset_name),
        'result_dataset_dir': os.path.join(experiment_dir, 'saved_results', args.model_name, args.dataset_name),
        'log_dataset_dir': os.path.join(experiment_dir, 'logs', args.model_name, args.dataset_name),
    }


def get_model_folder(args, model_name):
    paths = build_experiment_paths(args)
    return os.path.join(paths['model_dataset_dir'], model_name)


def get_result_folder(args):
    return build_experiment_paths(args)['result_dataset_dir']


def get_log_folder(args, run_name):
    paths = build_experiment_paths(args)
    return os.path.join(paths['log_dataset_dir'], run_name)


def infer_preprocessing_protocol(args):
    enabled = getattr(args, 'enabled_features', [])
    if not enabled:
        return 'base'
    feature_version = getattr(args, 'feature_version', 'v1')
    if 'trainfit' in feature_version:
        return 'trainfit'
    return 'legacy_full'


def build_result_metadata(args):
    return {
        'experiment_dir': getattr(args, 'experiment_dir', '') or '',
        'feature_bank_dir': getattr(args, 'feature_bank_dir', './processed_data'),
        'feature_version': getattr(args, 'feature_version', 'v1'),
        'preprocessing_protocol': infer_preprocessing_protocol(args),
        'enabled_features': list(getattr(args, 'enabled_features', [])),
        'feature_tag': getattr(args, 'feature_tag', ''),
    }
