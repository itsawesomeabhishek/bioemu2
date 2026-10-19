from functools import partial

from torch.nn import Linear

LinearNoBias = partial(Linear, bias=False)
