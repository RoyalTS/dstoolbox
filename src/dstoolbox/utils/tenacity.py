import typing
from tenacity import RetryCallState
from tenacity import _utils

if typing.TYPE_CHECKING:
    import loguru

    from tenacity import RetryCallState

def after_loguru(
    logger: "loguru._logger.Logger",
    log_level: typing.Union[int, str]
) -> typing.Callable[["RetryCallState"], None]:
    """After call strategy that logs to loguru logger the finished attempt."""

    def log_it(retry_state: "RetryCallState") -> None:
        callback_name = _utils.get_callback_name(retry_state.fn)
        logger.log(
            log_level,
            f"Finished call to '{callback_name}' after {retry_state.seconds_since_start:.2f} seconds, "
            f"this was the {_utils.to_ordinal(retry_state.attempt_number)} time calling it.",
        )

    return log_it
