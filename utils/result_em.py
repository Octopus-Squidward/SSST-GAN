import numpy as np

def result_em(EM_hat, M, A_hat, A_true):

    # E
    armse_em = np.mean(np.sqrt(np.mean((M - EM_hat) ** 2, axis=0)))
    p = A_true.shape[0]
    sad_err = np.zeros(p)
    for i in range(p):
        norm_EM_GT = np.sqrt(np.sum(M[:, i] ** 2, 0))
        norm_EM_hat = np.sqrt(np.sum(EM_hat[:, i] ** 2, 0))
        sad_err[i] = np.arccos(np.sum(EM_hat[:, i] * M[:, i].T, 0) / norm_EM_hat / norm_EM_GT)
    asad_em = np.mean(sad_err)

    # A
    p = A_true.shape[0]
    class_rmse = np.zeros(p)
    for i in range(p):
        class_rmse[i] = np.sqrt(np.mean((A_hat[i, :] - A_true[i, :]) ** 2, axis=0))
    armse = np.mean(class_rmse)
    armse_a = np.mean(np.sqrt(np.mean((A_hat - A_true) ** 2, axis=0)))

    return sad_err, asad_em, armse_em, class_rmse, armse, armse_a




