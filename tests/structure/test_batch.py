import torch

from bioemu2.structure.batch import pad_msa_features


def test_pad_msa_features() -> None:
    dummy_features = torch.ones(10, 5, 3)

    padded = pad_msa_features([dummy_features], max_pad=20, pad_value=0.0, pad_dim=0)[0]
    assert padded.shape == (20, 5, 3)
    torch.testing.assert_close(padded[:10], dummy_features)
    assert torch.all(padded[10:] == 0)

    padded = pad_msa_features([dummy_features], max_pad=20, pad_value=0.0, pad_dim=1)[0]
    assert padded.shape == (10, 20, 3)
    torch.testing.assert_close(padded[:, :5], dummy_features)
    assert torch.all(padded[:, 5:] == 0)

    padded = pad_msa_features([dummy_features], max_pad=20, pad_value=0.0, pad_dim=2)[0]
    assert padded.shape == (10, 5, 20)
    torch.testing.assert_close(padded[:, :, :3], dummy_features)
    assert torch.all(padded[:, :, 3:] == 0)
