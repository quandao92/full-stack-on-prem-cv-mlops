import os
import time
import pandas as pd
import numpy as np
from typing import List, Tuple, Union
from dvc.repo import Repo
from git import Git, GitCommandError
from prefect import task, get_run_logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from deepchecks.vision import classification_dataset_from_directory
from deepchecks.vision.suites import train_test_validation

@task(name='validate_data', log_prints=True)
def validate_data(ds_repo_path: str, save_path: str = 'ds_val.html', img_ext: str = 'jpeg'):
    logger = get_run_logger()
    train_ds, test_ds = classification_dataset_from_directory(
        root=os.path.join(ds_repo_path, 'images'), object_type='VisionData',
        image_extension=img_ext
    )
    suite = train_test_validation()
    logger.info("Running data validation test sute")
    result = suite.run(train_ds, test_ds)
    result.save_as_html(save_path)
    logger.info(f'Finish data validation and save report to {save_path}')
    logger.info("This file will also be saved along with the MLflow's training task in the later step")

@task(name='prepare_dvc_dataset')
def prepare_dataset(ds_root: str, ds_name: str, dvc_tag: str, dvc_checkout: bool = True):
    logger = get_run_logger()
    logger.info("Dataset name: {} | DvC tag: {}".format(ds_name, dvc_tag))
    ds_repo_path = os.path.join(ds_root, ds_name)

    annotation_path = os.path.join(ds_repo_path, 'annotation_df.csv')
    annotation_df = pd.read_csv(annotation_path)

    # check dvc_checkout field
    # if yes -> do git checkout, dvc pull, append path
    # if no -> warn and return path
    if dvc_checkout:
        git_repo = Git(ds_repo_path)
        try:
            git_repo.checkout(dvc_tag)
        except GitCommandError:
            valid_tags = git_repo.tag().split("\n")
            raise ValueError(f'Invalid dvc_tag. The tag might not exist. get {dvc_tag}. ' + \
                                f'existing tags: {valid_tags}')
        dvc_repo = Repo(ds_repo_path)
        logger.info('Running dvc diff to check whether files changed recently')
        logger.info('NOTE: There is an action needed after this command finishes. Please stay active.')
        start = time.time()
        result = dvc_repo.diff()
        end = time.time()
        logger.info(f'dvc diff took {end-start:.3f}s')
        if not result: # no change at all
            logger.info('The dataset does not have any modification.')
            ans = input('[ACTION] Do you still want to dvc checkout anyway? It might take some times. (yN)')
            if ans == 'y' or ans == 'Y':
                logger.info('Running dvc checkout...')
                start = time.time()
                dvc_repo.checkout()
                end = time.time()
                logger.info(f'Checkout completed. took {end-start:.3f}s')
        else:
            logger.info('Detected some modifications.')
            logger.info('Running dvc checkout...')
            start = time.time()
            dvc_repo.checkout()
            end = time.time()
            logger.info(f'Checkout completed. took {end-start:.3f}s')
    else:
        logger.warning(f'You set the dvc_checkout to false for {ds_name}. ' + \
                'The DvC will not check and pull the dataset repo, so please make sure they are correct.')
        
    return ds_repo_path, annotation_df






#======================CODE FOR TIME SERIES==============================================

@task(name='load_time_series_csv')
def load_time_series_csv(file_path: str, date_col: str, target_col: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Loading time series data from {file_path}...")

    data = pd.read_csv(file_path)
    if date_col not in data.columns or target_col not in data.columns:
        raise ValueError(f"Missing required columns: {date_col} or {target_col}")

    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(by=date_col)

    logger.info(f"Loaded data with shape {data.shape}")
    return data[[date_col, target_col]]


@task(name='prepare_time_series_data')
def prepare_time_series_data(data: pd.DataFrame, time_step: int, target_col: str) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    logger = get_run_logger()
    logger.info(f"Preparing time series data with time_step={time_step}...")

    if data[target_col].isnull().any():
        raise ValueError(f"Column {target_col} contains missing values.")
    if not np.issubdtype(data[target_col].dtype, np.number):
        raise ValueError(f"Column {target_col} must be numeric.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    target_values = scaler.fit_transform(data[[target_col]])

    X, y = [], []
    for i in range(len(target_values) - time_step):
        X.append(target_values[i:i + time_step, 0])
        y.append(target_values[i + time_step, 0])
    X, y = np.array(X), np.array(y)

    logger.info(f"Data prepared: X shape={X.shape}, y shape={y.shape}")
    return X, y, scaler



@task(name='split_time_series_data')
def split_time_series_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.1):
    logger = get_run_logger()
    logger.info("Splitting time series data...")

    if len(X) < 10:
        raise ValueError("Dataset is too small to split. Ensure it contains enough samples.")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, shuffle=False)
    val_split = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_split, shuffle=False)

    logger.info(f"Split completed: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test