import torch


@torch.jit.script
def atan2(y, x):
    pi = torch.pi
    ans = torch.arctan((y / x).float())
    ans = torch.where(
        x > 0,
        ans,
        torch.where(
            x != 0,
            torch.where(y >= 0, ans + pi, ans - pi),
            torch.where(
                y != 0,
                torch.where(
                    y > 0,
                    ans * 0 + pi / 2,
                    ans * 0 - pi / 2,
                ),
                ans * 0,
            ),
        ),
    )
    return ans


@torch.jit.script
def norm(x):
    m = x.mean(dim=0)
    return torch.sqrt(torch.sum((x - m) ** 2, dim=0))


@torch.jit.script
def normalize(x):
    return x / (norm(x) + 1e-16)


@torch.jit.script
def randomUnitVector(n: int):
    x = torch.randn(n)
    return normalize(x)


@torch.jit.script
def svd_1d(A):
    """The one-dimensional SVD"""
    assert A.shape == (2, 2)
    n_iter = 100

    x = randomUnitVector(2).to(dtype=A.dtype)
    B = torch.matmul(A, A.permute(1, 0))

    for _ in range(100):
        x = normalize(torch.matmul(B, x))
    return x


@torch.jit.script
def svd(A):
    assert A.shape == (2, 2)
    U = torch.zeros((2, 2), dtype=A.dtype)
    S = torch.zeros(2, dtype=A.dtype)
    V = torch.zeros((2, 2), dtype=A.dtype)

    matrixFor1D = A.clone()
    u = svd_1d(matrixFor1D)
    v_unnormalized = torch.matmul(A.permute(1, 0), u)
    sigma = norm(v_unnormalized)
    v = v_unnormalized / sigma
    U[0] = u
    S[0] = sigma
    V[0] = v

    matrixFor1D = A.clone()
    matrixFor1D -= sigma * torch.outer(u, v)
    u = svd_1d(matrixFor1D)
    v_unnormalized = torch.matmul(A.permute(1, 0), u)
    sigma = norm(v_unnormalized)
    v = v_unnormalized / sigma
    U[1] = u
    S[1] = sigma
    V[1] = v
    return U.permute(1, 0), S, V


@torch.jit.script
def svd2x2(A):
    N = A.shape[0]
    a = A[:, 0, 0]
    b = A[:, 0, 1]
    c = A[:, 1, 0]
    d = A[:, 1, 1]
    theta = 0.5 * atan2(2 * a * c + 2 * b * d, a * a + b * b - c * c - d * d)
    U = torch.zeros((N, 2, 2), dtype=A.dtype)
    U[:, 0, 0] = torch.cos(theta)
    U[:, 0, 1] = -torch.sin(theta)
    U[:, 1, 0] = torch.sin(theta)
    U[:, 1, 1] = torch.cos(theta)

    S1 = a * a + b * b + c * c + d * d
    S2 = torch.sqrt((a * a + b * b - c * c - d * d) ** 2 + 4 * (a * c + b * d) ** 2)
    sigma1 = torch.sqrt(S1 / 2 + S2 / 2)
    sigma2 = torch.sqrt(S1 / 2 - S2 / 2)
    S = torch.zeros((N, 2), dtype=A.dtype)
    S[:, 0] = sigma1
    S[:, 1] = sigma2
    phi = 0.5 * atan2(2 * a * b + 2 * c * d, a * a - b * b + c * c - d * d)
    W = torch.zeros((N, 2, 2), dtype=A.dtype)
    W[:, 0, 0] = torch.cos(phi)
    W[:, 0, 1] = -torch.sin(phi)
    W[:, 1, 0] = torch.sin(phi)
    W[:, 1, 1] = torch.cos(phi)
    s = torch.bmm(U.permute(0, 2, 1), torch.bmm(A, W))
    C = torch.zeros((N, 2, 2), dtype=A.dtype)
    C[:, 0, 0] = torch.sign(s[:, 0, 0])
    C[:, 1, 1] = torch.sign(s[:, 1, 1])
    V = torch.bmm(W, C)
    return U, S, V.permute(0, 2, 1)
