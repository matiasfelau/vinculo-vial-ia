import numpy as np

from vvi_ia.adapters.models.null import (
    NullCarSegmenter,
    NullCurbSegmenter,
    NullDamageDetector,
    NullEmbedder,
    NullFaceDetector,
    NullPlateReader,
    NullVehicleDetector,
)


def test_todos_los_puertos_nulos_devuelven_none_y_exponen_version() -> None:
    img = np.zeros((4, 4, 3), np.uint8)
    mask = np.zeros((4, 4), bool)

    assert NullVehicleDetector().detect(img) is None
    assert NullCarSegmenter().cars(img) is None
    assert NullCurbSegmenter().curb(img) is None
    assert NullEmbedder().embed(img, mask) is None
    assert NullPlateReader().read(img) is None
    assert NullDamageDetector().detect(img) is None
    assert NullFaceDetector().detect(img) is None
    for adapter in (
        NullVehicleDetector,
        NullCarSegmenter,
        NullCurbSegmenter,
        NullEmbedder,
        NullPlateReader,
        NullDamageDetector,
        NullFaceDetector,
    ):
        assert adapter.version == "null"
