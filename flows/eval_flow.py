import os
import yaml
import mlflow
import pandas as pd
from tasks.image_model import evaluate_model, load_saved_model
from tasks.timeseries_models import evaluate_timeseries_model, load_timeseries_model
from tasks.dataset import (
    prepare_dataset,
    load_time_series_csv,
    prepare_time_series_data,
    split_time_series_data
)
from flows.utils import log_mlflow_info, build_and_log_mlflow_url
from prefect import flow, get_run_logger
from prefect.artifacts import create_link_artifact
from typing import Dict, Any

CENTRAL_STORAGE_PATH = os.getenv("CENTRAL_STORAGE_PATH", "/home/ariya/central_storage")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@flow(name='eval_flow')
def eval_flow(cfg: Dict[str, Any], model_dir: str, model_metadata_file_path: str):
    logger = get_run_logger()
    data_type = cfg['evaluate']['data_type']  # Lấy loại dữ liệu
    eval_cfg = cfg['evaluate'][data_type]  # Chọn cấu hình theo loại dữ liệu
    mlflow_eval_cfg = eval_cfg['mlflow']

    logger.info('Preparing model for evaluation...')
    
    # Load model dựa trên loại dữ liệu
    if data_type == 'image':
        trained_model = load_saved_model(model_dir)
        model_cfg = None  # Không cần metadata cho image
    elif data_type == 'timeseries':
        # Hàm load_timeseries_model trả về cả model và metadata
        trained_model= load_timeseries_model(model_dir)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Đảm bảo rằng model_cfg đã được load cho timeseries
    with open(model_metadata_file_path, 'r') as f:
        model_cfg = yaml.safe_load(f)

    if data_type == 'image':
        # Cấu hình riêng cho image
        input_shape = (model_cfg['input_size']['h'], model_cfg['input_size']['w'])
        ds_repo_path, annotation_df = prepare_dataset(
            ds_root=cfg['dataset']['ds_root'], 
            ds_name=cfg['dataset']['ds_name'], 
            dvc_tag=cfg['dataset']['dvc_tag'], 
            dvc_checkout=cfg['dataset']['dvc_checkout']
        )
        evaluate_model(trained_model, model_cfg['classes'], ds_repo_path, annotation_df,
                       subset=eval_cfg['subset'], img_size=input_shape, classifier_type=model_cfg['classifier_type'])

    elif data_type == 'timeseries':
        # Cấu hình riêng cho timeseries
        data = load_time_series_csv(
            file_path=cfg['dataset']['file_path'], 
            date_col=cfg['dataset']['date_col'], 
            target_col=cfg['dataset']['target_col']
        )
        X, y, scaler = prepare_time_series_data(
            data=data, 
            time_step=cfg['dataset']['time_step'], 
            target_col=cfg['dataset']['target_col']
        )
        X_train, X_val, X_test, y_train, y_val, y_test = split_time_series_data(X=X, y=y)
        evaluate_timeseries_model(
            model=trained_model, 
            X_test=X_test, 
            y_test=y_test, 
            scaler=scaler, 
            # subset=eval_cfg['subset']
        )

    mlflow.set_experiment(mlflow_eval_cfg['exp_name'])
    with mlflow.start_run(description=mlflow_eval_cfg['exp_desc']) as eval_run:
        log_mlflow_info(logger, eval_run)
        eval_run_url = build_and_log_mlflow_url(logger, eval_run)
        mlflow.set_tags(tags=mlflow_eval_cfg['exp_tags'])
        mlflow.log_artifact(model_metadata_file_path)

    create_link_artifact(
        key='mlflow-evaluate-run',
        link=eval_run_url,
        description="Link to MLflow's evaluation run"
    )

def start(cfg):
    eval_cfg = cfg['evaluate']
    eval_flow(
        cfg=cfg,
        model_dir=eval_cfg['timeseries']['model_dir'],
        model_metadata_file_path=eval_cfg['timeseries']['model_metadata_file_path']
    )
