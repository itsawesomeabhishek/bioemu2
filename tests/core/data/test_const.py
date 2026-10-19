from bioemu2.data_interface.const import REF_ATOMS, Token, token_to_disto_atom_name


def test_tokens():
    for token in Token:
        if token != Token.PAD:
            assert len(token.singleletter) == 1
            if token != Token.GAP:
                assert token_to_disto_atom_name(token) in REF_ATOMS[token]
