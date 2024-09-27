def log_outputs(outputs: dict, stage: str, logger = None):
    s = f"Best {stage} Performance:"
    for key, value in outputs.items():
        s += f"{key} = {value:.4f} | "
    logger.info(s)



