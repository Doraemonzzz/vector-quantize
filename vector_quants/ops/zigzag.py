import torch


def zigzag_indices(rows, cols):
    indices = torch.zeros(rows * cols, dtype=torch.long)
    reverse_indices = torch.zeros(rows * cols, dtype=torch.long)
    r, c = 0, 0
    for i in range(rows * cols):
        # indices[i] = torch.tensor([r, c])
        j = r * cols + c
        indices[i] = j
        reverse_indices[j] = i
        if (r + c) % 2 == 0:
            if c == cols - 1:
                r += 1
            elif r == 0:
                c += 1
            else:
                r -= 1
                c += 1
        else:
            if r == rows - 1:
                c += 1
            elif c == 0:
                r += 1
            else:
                r += 1
                c -= 1
    return indices, reverse_indices
