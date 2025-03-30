import numpy as np
from matplotlib import pyplot as plt


def visualize_segmentation(
    image, num_classes, mask, prediction=None, prediction_set_size=None
):
    num_subplots = 0
    if prediction is not None:
        num_subplots += 1
    if prediction_set_size is not None:
        num_subplots += 1
    if mask is not None:
        num_subplots += 1

    if num_subplots == 0:
        raise ValueError(
            "At least one of mask, prediction or prediction_set_size must be provided."
        )

    num_classes = num_classes

    cmap = plt.cm.get_cmap("jet", num_classes)

    image = image / image.max()
    image = image - image.min()

    subplot = 1

    fig = plt.figure()

    if image.shape[0] > image.shape[1]:
        mode = "colwise"
    else:
        mode = "rowwise"

    mask = np.ma.masked_where(mask == 0, mask)

    if mode == "colwise":
        plt.subplot(1, num_subplots, subplot)
    else:
        plt.subplot(num_subplots, 1, subplot)

    if image.ndim == 2 or image.shape[-1] == 1:
        plt.imshow(image, cmap="gray")
    elif image.shape[-1] == 3:
        plt.imshow(image)
    else:
        raise ValueError(f"Image has invalid shape: {image.shape}")

    plt.imshow(
        mask, cmap=cmap, interpolation="none", alpha=0.5, vmin=1, vmax=num_classes
    )
    plt.grid(False)
    plt.axis("off")
    plt.title("Ground Truth")

    subplot += 1

    if prediction is not None:
        num_classes = prediction.shape[0]

        if mode == "colwise":
            plt.subplot(1, num_subplots, subplot)
        else:
            plt.subplot(num_subplots, 1, subplot)

        prediction = prediction.argmax(axis=0)

        prediction = np.ma.masked_where(prediction == 0, prediction)

        if image.ndim == 2 or image.shape[-1] == 1:
            plt.imshow(image, cmap="gray")
        elif image.shape[-1] == 3:
            plt.imshow(image)
        else:
            raise ValueError(f"Image has invalid shape: {image.shape}")

        plt.imshow(
            prediction,
            cmap=cmap,
            interpolation="none",
            alpha=0.2,
            vmin=1,
            vmax=num_classes,
        )
        plt.grid(False)
        plt.axis("off")
        plt.title("Prediction")

        subplot += 1

    if prediction_set_size is not None:
        uncertain = np.ma.masked_where(
            np.logical_or(prediction_set_size == 1, prediction == 0),
            prediction_set_size,
        )
        if mode == "colwise":
            plt.subplot(1, num_subplots, subplot)
        else:
            plt.subplot(num_subplots, 1, subplot)

        if image.ndim == 2 or image.shape[-1] == 1:
            plt.imshow(image, cmap="gray")
        elif image.shape[-1] == 3:
            plt.imshow(image)
        else:
            raise ValueError(f"Image has invalid shape: {image.shape}")

        plt.imshow(
            uncertain,
            cmap="Reds",
            interpolation="none",
            alpha=0.5,
            vmin=1,
            vmax=num_classes,
        )
        plt.grid(False)
        plt.axis("off")
        plt.title("Uncertainty")

    plt.tight_layout(pad=2.0)
    return fig
