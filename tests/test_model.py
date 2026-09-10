import torch

from avalanche_forecast.model import TemporalResNet50


def test_temporal_resnet_produces_one_logit_per_sample():
    model = TemporalResNet50(image_channels=2, meteo_features=3, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 2, 2, 64, 64), torch.randn(1, 2, 3))
    assert output.shape == (1,)
