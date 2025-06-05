# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops, DEFAULT_CFG


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    # def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, save_dir="D:/PycharmProject/detectProject/"):
    #     """
    #     Initializes the DetectionPredictor class.
    #
    #     Args:
    #         cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
    #         overrides (dict, optional): Configuration overrides. Defaults to None.
    #         save_dir (str, optional): Directory to save results. Defaults to "your/default/save/path/here".
    #     """
    #     super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks, save_dir=save_dir)

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results


import argparse


def main():
    parser = argparse.ArgumentParser(description="Run YOLO prediction on an image with specified save directory.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to the YOLO model file")
    parser.add_argument("--source", type=str,
                        default="D:/PycharmProject/detectProject/ultralytics-main/ultralytics/assets/zidane.jpg",
                        help="Path to the input image")
    parser.add_argument("--imgsz", type=int, default=640, help="images size for inference")
    parser.add_argument("--show", default=True, action="store_true", help="Whether to display the result image")
    parser.add_argument("--save", default=True, action="store_true", help="Whether to save the result image")

    args = parser.parse_args()

    # Construct arguments dictionary
    args_dict = vars(args)

    # Create DetectionPredictor instance with specified save directory
    predictor = DetectionPredictor(overrides=args_dict)

    # Run prediction
    predictor.predict_cli()


if __name__ == "__main__":
    main()
