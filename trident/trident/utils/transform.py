import numpy as np
import torch

# from torch.nn.utils.rnn import pad_sequence

# N = 30
# L = 8
# tensors = [
#     torch.randn((N, torch.randint(low=10, high=20, size=(1,)), L)) for x in range(10)
# ]
# for t in tensors:
#     print(t.shape)
# pad_sequence(tensors)


def stack_or_pad_2d(tensors: list[torch.Tensor], pad_id=-100) -> torch.Tensor:
    """
    Stack along first axis of latter axis is homogenous in length else pad and stack.
    """
    N, D = zip(*[tuple(x.shape) for x in tensors])
    if len(set(D)) != 1:
        out = torch.full_like(
            torch.Tensor(sum(N), max(D)), fill_value=-100, device=tensors[0].device
        )
        start = 0
        for t in tensors:
            num, len_ = t.shape
            out[start : start + num, :len_] = t
            start += num
        return out
    return torch.vstack(tensors)


def concatenate_3d(tensors: list[torch.Tensor], pad_id=-100) -> torch.Tensor:
    # (N sequences, L individual sequence length, C num classes -- typically)
    N, L, C = zip(*[tuple(x.shape) for x in tensors])
    out = torch.full_like(
        torch.Tensor(sum(N), max(L), max(C)), fill_value=-100, device=tensors[0].device
    )
    start = 0
    for t in tensors:
        num, len_, _ = t.shape
        out[start : start + num, :len_, :] = t
        start += num
    return out


def flatten_dict(inputs: list[dict]) -> dict:
    """Conflates keys of list[dict] and stacks np arrays & tensors along 0-dim."""
    ret = {}
    for input_ in inputs:
        for k, v in input_.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        if isinstance(v[0], torch.Tensor):
            dim = v[0].dim()
            # stack matrices along first axis
            if dim == 2:
                ret[k] = stack_or_pad_2d(v)
            # concatenate vectors
            elif dim == 1:
                ret[k] = torch.cat(v, dim=0)
            # pad varying dimension and concatenate
            elif dim == 3:
                ret[k] = concatenate_3d(v)
            else:
                raise NotImplementedError(
                    f"Handling {dim} number of dimensions unimplemented"
                )
        elif isinstance(v[0], np.ndarray):
            ret[k] = np.vstack(v)
        else:
            pass
    return ret
