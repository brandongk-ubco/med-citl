import neptune
from . import cli
from loguru import logger
from .dataset import Dataset
import onnxruntime
from tqdm import tqdm
import numpy as np
from torchmetrics.classification import JaccardIndex
import torch


def softmax(x):
    """Compute softmax values along the first axis (index 0) of a 3D array."""
    # Keep the max along axis 0 for numerical stability
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)


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

    jaccard = JaccardIndex(
        task="multiclass",
        num_classes=datamodule.num_classes,
        average="none",
        ignore_index=0,
        zero_division=1.0,
    )

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    predictions = []
    ground_truths = []

    dataloader = datamodule.test_dataloader()

    predictions = np.empty(
        (len(dataloader), datamodule.num_classes, 1024, 2048), dtype=np.float16
    )
    ground_truths = np.empty((len(dataloader), 1024, 2048), dtype=np.uint8)

    i = 0

    for batch in tqdm(datamodule.test_dataloader(), desc="Inferencing"):
        inputs, targets, _index = batch

        batch_size = inputs.shape[0]

        for idx in range(batch_size):
            input = inputs[idx]
            target = targets[idx]

            input_np = input.numpy()
            prediction_0 = ort_session.run(
                [output_name], {input_name: input_np[:, :512, :1024]}
            )
            prediction_1 = ort_session.run(
                [output_name], {input_name: input_np[:, :512, 1024:]}
            )
            prediction_2 = ort_session.run(
                [output_name], {input_name: input_np[:, 512:, :1024]}
            )
            prediction_3 = ort_session.run(
                [output_name], {input_name: input_np[:, 512:, 1024:]}
            )

            stitched_prediction = np.zeros(
                (datamodule.num_classes, input_np.shape[1], input_np.shape[2]),
                dtype=prediction_0[0].dtype,
            )

            stitched_prediction[:, :512, :1024] = prediction_0[0]
            stitched_prediction[:, :512, 1024:] = prediction_1[0]
            stitched_prediction[:, 512:, :1024] = prediction_2[0]
            stitched_prediction[:, 512:, 1024:] = prediction_3[0]

            prediction = softmax(stitched_prediction)
            jaccard(
                torch.Tensor(prediction).unsqueeze(0), torch.Tensor(target).unsqueeze(0)
            )
            predictions[i, :, :, :] = stitched_prediction.astype(np.float16)
            ground_truths[i, :, :] = target.numpy()

            i += 1

    np.savez_compressed(
        f"{run_id}.npz",
        predictions=np.stack(predictions),
        ground_truths=np.stack(ground_truths),
    )

    jaccards = dict(
        zip(datamodule.classes, [n * 100 for n in jaccard.compute().tolist()])
    )
    logger.info(f"IoU: {jaccards}")
    logger.info(f"Mean IoU: {jaccard.compute()[1:].mean() * 100:.2f}")
