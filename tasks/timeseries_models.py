import os
import shutil
import hashlib
import yaml
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from prefect import task, get_run_logger
import mlflow
import matplotlib.pyplot as plt


@task(name="build_timeseries_model")
def build_timeseries_model(input_shape, model_type="LSTM", lstm_units=128, conv_filters=128, kernel_size=3, dropout_rate=0.2):
    """
    Xây dựng mô hình time series dựa trên loại được chỉ định.
    """
    logger = get_run_logger()
    logger.info(f"Building {model_type} model with input shape {input_shape}...")

    if model_type == "LSTM":
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(lstm_units),
            Dropout(dropout_rate),
            Dense(1)
        ])
    elif model_type == "GRU":
        model = Sequential([
            GRU(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            GRU(lstm_units // 2),
            Dropout(dropout_rate),
            Dense(1)
        ])
    elif model_type == "BiLSTM":
        model = Sequential([
            Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape),
            Dropout(dropout_rate),
            Bidirectional(LSTM(lstm_units // 2)),
            Dropout(dropout_rate),
            Dense(1)
        ])
    elif model_type == "Conv1D-BiLSTM":
        model = Sequential([
            Conv1D(conv_filters, kernel_size=kernel_size, activation="relu", input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Bidirectional(LSTM(lstm_units, return_sequences=True)),
            Dropout(dropout_rate),
            Bidirectional(LSTM(lstm_units // 2)),
            Dropout(dropout_rate),
            Dense(1)
        ])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    logger.info(f"{model_type} model built successfully.")
    return model


@task(name="train_timeseries_model")
def train_timeseries_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, patience=10):
    """
    Huấn luyện mô hình time series.
    """
    logger = get_run_logger()
    logger.info(f"Starting training with {epochs} epochs and batch size {batch_size}...")

    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=2
    )

    # Log hyperparameters vào MLflow
    if mlflow.active_run() is None:
        mlflow.start_run()
    mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "patience": patience})

    # Log loss và val_loss cho từng epoch
    for epoch, (train_loss, val_loss) in enumerate(zip(history.history["loss"], history.history["val_loss"])):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    # Log giá trị loss cuối cùng
    mlflow.log_metrics({
        "final_train_loss": history.history["loss"][-1],
        "final_val_loss": history.history["val_loss"][-1]
    })

    logger.info("Training completed successfully.")
    return model, history



@task(name="evaluate_timeseries_model")
def evaluate_timeseries_model(model, X_test, y_test, scaler):
    """
    Đánh giá mô hình time series.

    Args:
        model: Mô hình đã được huấn luyện.
        X_test: Dữ liệu đầu vào kiểm tra.
        y_test: Nhãn thật của dữ liệu kiểm tra.
        scaler: Bộ scaler để đảo ngược giá trị dự đoán và nhãn thật.

    Returns:
        mse: Mean Squared Error của dự đoán.
        mae: Mean Absolute Error của dự đoán.
    """
    logger = get_run_logger()
    logger.info("Evaluating the time-series model...")

    try:
        # Dự đoán giá trị từ mô hình
        y_pred = model.predict(X_test)
        logger.info("Prediction completed.")

        # Đảo ngược scaler để về không gian ban đầu
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        logger.info("Inverse transformation completed.")

        # Tính toán các chỉ số hiệu suất
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"Mean Squared Error (MSE): {mse}")
        logger.info(f"Mean Absolute Error (MAE): {mae}")

        # Vẽ biểu đồ so sánh True vs Predicted
        logger.info("Creating comparison plot...")
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label="True Values", alpha=0.7)
        plt.plot(y_pred, label="Predicted Values", alpha=0.7)
        plt.legend()
        plt.title("True vs Predicted Values")
        plt.xlabel("Time Steps")
        plt.ylabel("Values")

        # Lưu biểu đồ
        plot_path = "evaluation_timeseries_plot.png"
        # plt.savefig(plot_path)
        plt.close()
        logger.info(f"Comparison plot saved at {plot_path}")

        # Log biểu đồ vào MLflow nếu có active run
        if mlflow.active_run() is None:
            mlflow.start_run()
            
        if mlflow.active_run():
            try:
                mlflow.log_artifact(plot_path)
                mlflow.log_metric("MSE", mse)
                mlflow.log_metric("MAE", mae)
                logger.info(f"Plot and metrics logged to MLflow: {plot_path}")
            except Exception as e:
                logger.error(f"Failed to log to MLflow: {str(e)}")

        return mse, mae

    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise

@task(name="save_timeseries_model")
def save_timeseries_model(model, model_cfg: dict):
    """
    Lưu mô hình và metadata vào thư mục chỉ định.
    """
    logger = get_run_logger()

    # Đường dẫn lưu mô hình
    model_dir = os.path.join(model_cfg['save_dir'], model_cfg['model_name'])
    metadata_path = os.path.join(model_cfg['save_dir'], f"{model_cfg['model_name']}.yaml")

    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(model_cfg['save_dir']):
        os.makedirs(model_cfg['save_dir'])

    # Lưu mô hình
    model.save(model_dir)
    logger.info(f"Model saved to {model_dir}")

    # Lưu metadata
    metadata = model_cfg.copy()
    metadata.pop('save_dir', None)
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)
    logger.info(f"Metadata saved to {metadata_path}")

    # Tính toán hash của mô hình để kiểm tra sự thay đổi
    model_hash = calculate_model_hash(model_dir)
    logger.info(f"Model hash: {model_hash}")

    return model_dir, metadata_path, model_hash


def calculate_model_hash(model_dir):
    """
    Tính toán hash cho toàn bộ thư mục mô hình.
    """
    hasher = hashlib.md5()
    for root, _, files in os.walk(model_dir):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
    return hasher.hexdigest()


@task(name="upload_timeseries_model")
def upload_timeseries_model(model_dir: str, metadata_file_path: str, remote_dir: str):
    """
    Tải mô hình và metadata lên thư mục từ xa (hoặc lưu trữ trên cloud).
    """
    logger = get_run_logger()
    model_name = os.path.basename(model_dir)

    # Tạo thư mục từ xa nếu chưa tồn tại
    if not os.path.exists(remote_dir):
        os.makedirs(remote_dir)

    # Copy mô hình
    remote_model_dir = os.path.join(remote_dir, model_name)
    shutil.copytree(model_dir, remote_model_dir, dirs_exist_ok=True)
    logger.info(f"Model uploaded to {remote_model_dir}")

    # Copy metadata
    remote_metadata_path = os.path.join(remote_dir, os.path.basename(metadata_file_path))
    shutil.copy2(metadata_file_path, remote_metadata_path)
    logger.info(f"Metadata uploaded to {remote_metadata_path}")

    return remote_model_dir, remote_metadata_path


@task(name="load_timeseries_model")
def load_timeseries_model(model_path: str):
    """
    Tải lại mô hình và đọc metadata.
    """
    logger = get_run_logger()

    # Tải mô hình
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = load_model(model_path)
    logger.info(f"Model loaded from {model_path}")

    # # Đọc metadata
    # if not os.path.exists(metadata_file_path):
    #     raise FileNotFoundError(f"Metadata file not found at {metadata_file_path}")
    # with open(metadata_file_path, 'r') as f:
    #     metadata = yaml.safe_load(f)
    # logger.info(f"Metadata loaded from {metadata_file_path}")

    return model
