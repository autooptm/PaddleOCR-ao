# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

from .logging import logger

ENV_SWITCH = "PADDLEOCR_OPT_1"


def _enabled():
    return os.environ.get(ENV_SWITCH, "1").strip().lower() not in ("0", "false", "off", "no")


def opt_3(paddlex_pipeline):
    """Install the device-side CTC reduction on an OCR pipeline's text recognizer.

    Returns True when installed. Anything unexpected (CPU device, another engine,
    another runner, a model without a CTC head) leaves the stock path untouched."""
    if not _enabled():
        return False
    try:
        pipeline = getattr(paddlex_pipeline, "_pipeline", paddlex_pipeline)
        rec = getattr(pipeline, "text_rec_model", None)
        runner = getattr(rec, "runner", None)
        post_op = getattr(rec, "post_op", None)
        if runner is None or post_op is None or not hasattr(post_op, "decode"):
            return False
        if (getattr(runner, "_config", None) or {}).get("device_type") != "gpu":
            return False
        if str((getattr(runner, "_config", None) or {}).get("run_mode", "paddle")).startswith("trt"):
            return False
        predictor = getattr(runner, "predictor", None)
        if predictor is None:
            predictor = runner.infer.predictor
        import numpy as np
        import paddle

        place = paddle.CUDAPlace(int(runner._config.get("device_id") or 0))
    except Exception as e:  # noqa: BLE001
        logger.debug("optimized CTC path not installed: %s", e)
        return False

    def opt_7(x):
        inputs = [
            paddle.to_tensor(np.ascontiguousarray(a), place=place) for a in x
        ]
        probs = predictor.run(inputs)[0]
        return _Opt5(
            paddle.argmax(probs, axis=-1).numpy(), paddle.max(probs, axis=-1).numpy()
        )

    rec.runner = opt_7
    rec.post_op = _Opt6(post_op)
    return True


class _Opt5:
    __slots__ = ("idx", "prob")

    def __init__(self, idx, prob):
        self.idx, self.prob = idx, prob


class _Opt6:

    def __init__(self, stock):
        self.stock = stock

    def __call__(self, pred, return_word_box=False, **kwargs):
        if not isinstance(pred, _Opt5):
            return self.stock(pred, return_word_box=return_word_box, **kwargs)
        text = self.stock.decode(
            pred.idx,
            pred.prob,
            is_remove_duplicate=True,
            return_word_box=return_word_box,
        )
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs["wh_ratio_list"][rec_idx]
                max_wh_ratio = kwargs["max_wh_ratio"]
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)
        texts = [t[0] if len(t) <= 2 else (t[0], t[2]) for t in text]
        scores = [t[1] for t in text]
        return texts, scores
