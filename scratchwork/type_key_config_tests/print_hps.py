import determined as det

if __name__ == "__main__":
    hparams = det.get_cluster_info().trial.hparams
    print(hparams)
