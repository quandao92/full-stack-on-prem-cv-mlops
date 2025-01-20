import os
import yaml
import shutil
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from prefect import task, get_run_logger
from prefect.artifacts import create_link_artifact
from typing import List, Dict, Union
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from .utils.tf_data_utils import build_data_pipeline
from .utils.callbacks import MLflowLog



def build_classification_report_df(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return pd.DataFrame(report).T

# build metadata for using along with the model in deployment
def build_model_metadata(model_cfg): 
    metadata = model_cfg.copy()
    metadata.pop('save_dir')
    return metadata

def build_figure_from_df(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    table = pd.plotting.table(ax, df, loc='center', cellLoc='center')  # where df is your data frame
    plt.show()
    return fig, table

@task(name='upload_model')
def upload_model(model_dir: str, metadata_file_path: str, remote_dir: str):
    # this is the step you should replace with uploading the file
    # to a cloud storage if you want to deploy on cloud
    logger = get_run_logger()
    model_name = os.path.split(model_dir)[-1]
    metadata_file_name = os.path.split(metadata_file_path)[-1]

    shutil.copy2(metadata_file_path, remote_dir)
    
    model_save_dir = os.path.join(remote_dir, model_name)
    shutil.copytree(model_dir, model_save_dir, dirs_exist_ok=True)
    logger.info(f'Uploaded the model & the metadata file from {model_save_dir}')
    return model_save_dir, metadata_file_name

@task(name='load_model')
def load_saved_model(model_path: str):
    logger = get_run_logger()
    logger.info(f'Loading the model from {model_path}')
    model = load_model(model_path)
    logger.info('Loaded successfully')
    return model


@task(name='save_model')
def save_model(model: tf.keras.models.Model, model_cfg: Dict[str, Union[str, List[str], List[int]]]):
    import hashlib

    def calculate_model_hash(model_dir):
        """Tính toán hash cho mô hình để kiểm tra xem mô hình đã thay đổi hay chưa."""
        hasher = hashlib.md5()
        for root, _, files in os.walk(model_dir):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
        return hasher.hexdigest()

    logger = get_run_logger()

    # Bắt đầu một run nếu chưa có
    if mlflow.active_run() is None:
        mlflow.start_run()
        logger.info("Started a new MLflow run.")

    # Lấy thông tin Run ID
    run_id = mlflow.active_run().info.run_id
    logger.info(f"Active Run ID: {run_id}")

    # Lưu mô hình vào thư mục
    model_dir = os.path.join(model_cfg['save_dir'], model_cfg['model_name'])
    if not os.path.exists(model_cfg['save_dir']):
        os.makedirs(model_cfg['save_dir'])

    # Compile mô hình trước khi lưu
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.save(model_dir)
    logger.info(f"Model is saved to {model_dir}")

    # Lưu metadata của mô hình
    model_metadata = build_model_metadata(model_cfg)
    metadata_save_path = os.path.join(model_cfg['save_dir'], model_cfg['model_name']+'.yaml')
    with open(metadata_save_path, 'w') as f:
        yaml.dump(model_metadata, f)

    # Log artifacts vào MLflow
    mlflow.log_artifact(model_dir)
    mlflow.log_artifact(metadata_save_path)

    # Tính toán hash của mô hình
    current_model_hash = calculate_model_hash(model_dir)
    if 'model_hash' in model_cfg and model_cfg['model_hash'] == current_model_hash:
        logger.info("Model has not changed. Skipping version registration.")
        return model_dir, metadata_save_path

    # Lưu hash hiện tại vào model_cfg
    model_cfg['model_hash'] = current_model_hash

    # Đường dẫn Artifact URI
    model_uri = f"runs:/{run_id}/{model_cfg['model_name']}"
    model_name = model_cfg['model_name']
    
    client = MlflowClient()
    
    # Kiểm tra xem phiên bản đã tồn tại hay chưa
    existing_versions = client.search_model_versions(f"name='{model_name}'")
    for version in existing_versions:
        if version.source == model_uri and version.run_id == run_id:
            logger.info(f"Model version already exists: Version {version.version}")
            return model_dir, metadata_save_path
    
    # Đăng ký mô hình nếu chưa có
    try:
        if not existing_versions:
            registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            logger.info(f"Model registered with name: {registered_model.name}")
        
        # Tạo phiên bản nếu cần
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        logger.info(f"Model version {model_version.version} registered successfully!")
    except Exception as e:
        logger.error(f"Error during model registration: {str(e)}")
    
    # Cập nhật metadata và stage cho phiên bản
    try:
        # description = (
        #     f"Version {model_version.version} of the classification model. "
        #     f"Trained with dataset: {model_cfg.get("dataset_name", model_metadata.get("dataset", "DefaultDataset"))}, "
        #     f"epochs: {model_cfg.get("epochs", model_metadata.get("epochs", 10))}, "
        #     f"learning rate: {model_cfg.get("learning_rate", model_metadata.get("learning_rate", 0.001))}."
        # )
        description=(
            f"Version {model_version.version} of the classification model. "
            f"Trained with dataset: {model_cfg.get('dataset_name', model_metadata.get('dataset', 'DefaultDataset'))}, "
            f"epochs: {model_cfg.get('epochs', model_metadata.get('epochs', 10))}, "
            f"learning rate: {model_cfg.get('learning_rate', model_metadata.get('learning_rate', 0.001))}."
        )
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description
        )
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.info(f"Model version {model_version.version} transitioned to stage 'Staging'.")
        
        # Thêm các thẻ động dựa trên model_cfg
        tags = {
            "framework": "TensorFlow",
            "dataset": model_cfg.get("dataset_name", model_metadata.get("dataset", "DefaultDataset")),
            "epochs": str(model_cfg.get("epochs", model_metadata.get("epochs", 10))),
            "learning_rate": str(model_cfg.get("learning_rate", model_metadata.get("learning_rate", 0.001))),
            "training_time": model_cfg.get("training_time", model_metadata.get("training_time", "Not specified")),
            "accuracy": model_cfg.get("accuracy", model_metadata.get("accuracy", "Not computed")),
        }

        # Gắn từng thẻ vào phiên bản
        for key, value in tags.items():
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key=key,
                value=value
            )
        logger.info(f"Updated model version {model_version.version} with tags: {tags}")
    except Exception as e:
        logger.error(f"Error during updating model version metadata: {str(e)}")
    
    return model_dir, metadata_save_path


    
@task(name='build_model')
def build_model(input_size: list, n_classes: int, classifier_activation: str = 'softmax',
                classification_layer: str = 'classify'):
    logger = get_run_logger()
    # backbone = ResNet50(include_top=False, weights='imagenet',
    #                      input_shape = [input_size[0], input_size[1], 3])
    # x = GlobalAveragePooling2D()(backbone.output)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dense(n_classes, activation=classifier_activation)(x)

    backbone = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', 
                               input_shape=[input_size[0],input_size[1], 3]),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
    ])

    x = GlobalAveragePooling2D()(backbone.output)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(n_classes, activation=classifier_activation, name=classification_layer)(x)
    model = Model(inputs=backbone.input, outputs=x)
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    logger.info(f"Model summary:")
    logger.info('\n'.join(summary))
    return model
                
@task(name='train_model')
def train_model(model: tf.keras.models.Model, classes: List[str], ds_repo_path: str, 
                annotation_df: pd.DataFrame, img_size: List[int], epochs: int, batch_size: int, 
                init_lr: float, augmenter: iaa):
    logger = get_run_logger()
    logger.info('Building data pipelines')
    
    # Log thông tin dataset vào MLflow
    if mlflow.active_run() is None:
        mlflow.start_run()
    mlflow.log_param("dataset", ds_repo_path)
    mlflow.log_param("num_classes", len(classes))
    mlflow.log_param("img_size", img_size)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", init_lr)
    
    train_ds = build_data_pipeline(annotation_df, classes, 'train', img_size, batch_size, 
                                   do_augment=True, augmenter=augmenter)
    valid_ds = build_data_pipeline(annotation_df, classes, 'valid', img_size, batch_size, 
                                   do_augment=False, augmenter=None)
    # compile
    opt = Adam(learning_rate=init_lr)
    loss = CategoricalCrossentropy()
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    # callbacks
    mlflow_log = MLflowLog()
    
    # fit
    logger.info('Start training')
    model.fit(train_ds,
              validation_data=valid_ds,
              epochs=epochs,
              callbacks=[mlflow_log]
             )
    
    # return trained model
    return model

@task(name='evaluate_model')
def evaluate_model(model: tf.keras.models.Model, classes: List[str], ds_repo_path: str, 
                   annotation_df: pd.DataFrame, subset: str, img_size: List[int], classifier_type: str='multi-class', 
                   multilabel_thr: float=0.5):
    logger = get_run_logger()
    
    # Mở một MLflow run nếu chưa có
    if mlflow.active_run() is None:
        mlflow.start_run()
        logger.info("Started a new MLflow run.")

    # Log dataset và thông tin cấu hình
    mlflow.log_param("dataset", ds_repo_path)
    mlflow.log_param("subset", subset)
    mlflow.log_param("img_size", img_size)
    mlflow.log_param("classifier_type", classifier_type)
    if classifier_type == 'multi-label':
        mlflow.log_param("multilabel_threshold", multilabel_thr)
    
    logger.info(f"Building a data pipeline from '{subset}' set")
    test_ds = build_data_pipeline(annotation_df, classes, subset, img_size,
                                   do_augment=False, augmenter=None)
    logger.info('Getting ground truths and making predictions')
    y_true_bin = np.concatenate([y for _, y in test_ds], axis=0)
    y_pred_prob = model.predict(test_ds)
    if classifier_type == 'multi-class':
        y_true = np.argmax(y_true_bin, axis=1)
        y_pred = tf.argmax(y_pred_prob, axis=1)
    else: # multi-label
        y_true = y_true_bin
        y_pred = (y_pred_prob > multilabel_thr).astype(np.int8)

    if classifier_type == 'multi-class':
        # Create a confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Plot the confusion matrix
        conf_matrix_fig = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True,
                   xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Calculate AUC
        roc_auc = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr')
        
        # Print classification report
        report = build_classification_report_df(y_true, y_pred, classes)
        
    elif classifier_type == 'multi-label':
        conf_matrix_fig = None
        roc_auc = roc_auc_score(y_true, y_pred_prob, average=None, multi_class='ovr')
        
        # Print classification report
        report = build_classification_report_df(y_true, y_pred, classes)
        report['AUC'] = list(roc_auc) + (4*[None])
    logger.info('Logging outputs to MLflow to finish the process')
    if conf_matrix_fig:
        mlflow.log_figure(conf_matrix_fig, 'confusion_matrix.png')
    if isinstance(roc_auc, float):
        mlflow.log_metric("AUC", roc_auc)
    # log_figure is a lot easier to look at from ui than log_table
    report = report.apply(lambda x: round(x, 5))
    report = report.reset_index()
    report_fig, _ = build_figure_from_df(report)
    mlflow.log_figure(report_fig, 'classification_report.png')
    mlflow.log_table(report, 'classification_report.json')
    