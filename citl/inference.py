import neptune
from . import cli
from loguru import logger
from .dataset import Dataset
import pandas as pd
import onnxruntime
from tqdm import tqdm
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


@cli.command()
def inference(run_id: str, dataset: Dataset):
    with neptune.init_run(with_id=run_id, mode="read-only") as run:
        artifact_path = "model.onnx"

        logger.info(f"Accessing artifact: {artifact_path}")
        artifact = run[artifact_path]

        logger.info("Fetching ONNX model content into memory...")
        artifact.download()
        logger.info("ONNX model content fetched successfully.")
        ort_session = onnxruntime.InferenceSession(
            "model.onnx", providers=["CUDAExecutionProvider"]
        )

    datamodule = Dataset.get(dataset)()
    datamodule.setup(stage="test")

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    rows = []

    for batch in tqdm(datamodule.test_dataloader(), desc="Inferencing"):
        inputs, targets, _index = batch

        batch_size = inputs.shape[0]

        for idx in range(batch_size):
            input = inputs[idx]
            target = targets[idx]

            input_np = input.numpy()
            prediction = ort_session.run([output_name], {input_name: input_np})[
                0
            ].squeeze()
            prediction = softmax(prediction).tolist()
            row = dict(zip(datamodule.classes, prediction))
            row["label"] = int(target.numpy())
            row["predicted"] = prediction.index(max(prediction))
            row["correct"] = row["label"] == row["predicted"]
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("predictions_pandas.csv", index=False)
    print("Predictions (with pandas) saved to predictions_pandas.csv")

    accuracy = (df["correct"].sum() / len(df)) * 100
    logger.info(f"Accuracy: {accuracy:.2f}")
