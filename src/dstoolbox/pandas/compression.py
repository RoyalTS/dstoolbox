"""Compress pandas DataFrames for better memory footprint."""
import numpy as np
import pandas as pd
from loguru import logger

INT8_MIN = np.iinfo(np.int8).min
INT8_MAX = np.iinfo(np.int8).max
INT16_MIN = np.iinfo(np.int16).min
INT16_MAX = np.iinfo(np.int16).max
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max

FLOAT16_MIN = np.finfo(np.float16).min
FLOAT16_MAX = np.finfo(np.float16).max
FLOAT32_MIN = np.finfo(np.float32).min
FLOAT32_MAX = np.finfo(np.float32).max


def _total_memory_usage(df: pd.DataFrame):
    """Compute total memory used."""
    logger.trace(df.memory_usage())

    memory = df.memory_usage().sum() / (1024 * 1024)
    logger.debug(f"Memory usage : {memory:.2f}MB")

    return memory


def compress_dataset(
    df: pd.DataFrame,
    min_int_bits: int = 32,
    min_float_bits: int = 32,
) -> pd.DataFrame:
    """Compress df by downcasting each column to the smallest possible int/float type that will hold the column's contents.

    Parameters
    ----------
    df: pandas.DataFrame

    Returns
    -------
        pandas.DataFrame
    """
    df = df.copy()
    memory_before_compress = _total_memory_usage(df)

    for col in df.select_dtypes(exclude=["object", "category"]).columns:
        col_dtype = df[col].dtype

        logger.trace(f"Name: {col}, Type: {col_dtype}")
        col_series = df[col]
        col_min = col_series.min()
        col_max = col_series.max()

        if col_dtype == "float64":
            logger.trace(f"  variable min: {col_min:.4f} max: {col_max:.4f}")
            if (
                (min_float_bits <= 16)
                and (FLOAT16_MIN < col_min)
                and (col_max < FLOAT16_MAX)
            ):
                df[col] = df[col].astype(np.float16)
                logger.trace(
                    f"    float16 min: {FLOAT16_MIN:.4f} max: {FLOAT16_MAX:.4f}",
                )
                logger.trace("    compress float64 ==> float16")
            elif (
                (min_float_bits <= 32)
                and (FLOAT32_MIN < col_min)
                and (col_max < FLOAT32_MAX)
            ):
                df[col] = df[col].astype(np.float32)
                logger.trace(
                    f"    float32 min: {FLOAT32_MIN:.4f} max: {FLOAT32_MAX:.4f}",
                )
                logger.trace("    compress float64 ==> float32")
            else:
                pass

        if col_dtype == "int64":
            logger.trace(f"  variable min: {col_min:d} max: {col_max:d}")
            type_flag = 64
            if (
                (min_int_bits <= 8)
                and (INT8_MIN / 2 < col_min)
                and (col_max < INT8_MAX / 2)
            ):
                type_flag = 8
                df[col] = df[col].astype(np.int8)
                logger.trace(f"    int8 min: {INT8_MIN:d} max: {INT8_MAX:d}")
            elif (
                (min_int_bits <= 16) and (INT16_MIN < col_min) and (col_max < INT16_MAX)
            ):
                type_flag = 16
                df[col] = df[col].astype(np.int16)
                logger.trace(f"    int16 min: {INT16_MIN:d} max: {INT16_MAX:d}")
            elif (
                (min_int_bits <= 32) and (INT32_MIN < col_min) and (col_max < INT32_MAX)
            ):
                type_flag = 32
                df[col] = df[col].astype(np.int32)
                logger.trace(f"    int32 min: {INT32_MIN:d} max: {INT32_MAX:d}")
                type_flag = 1
            else:
                pass

            if type_flag == 32:
                logger.trace("    compress (int64) ==> (int32)")
            elif type_flag == 16:
                logger.trace("    compress (int64) ==> (int16)")
            else:
                logger.trace("    compress (int64) ==> (int8)")

    memory_after_compress = _total_memory_usage(df)
    compression_rate = 1 - memory_after_compress / memory_before_compress
    logger.debug(f"Compression Rate: {compression_rate:.2%}")

    return df
