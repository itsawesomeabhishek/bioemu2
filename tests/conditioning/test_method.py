from bioemu2.conditioning.method import num_method_types


def test_checkpoint_method_embedding_size() -> None:
    # The trained checkpoint's method-conditioning embedding weights have 17 rows;
    # changing this size prevents those parameters from loading.
    assert num_method_types == 17
