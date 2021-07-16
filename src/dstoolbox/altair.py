from loguru import logger


def sample_for_altair(df):
    """Sample a pandas DataFrame for altair plotting if necessary."""
    if len(df) > 5000:
        logger.warning(f"Sampling down dataset from {len(df)} to 5000 rows")
        return df.sample(5000)
    else:
        return df
