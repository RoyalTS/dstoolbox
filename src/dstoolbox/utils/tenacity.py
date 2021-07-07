import typing

from tenacity import RetryCallState, _utils
from tenacity.compat import get_exc_info_from_future

if typing.TYPE_CHECKING:
    import loguru
    from tenacity import RetryCallState


def before_sleep_loguru(
    logger: "loguru._logger.Logger",
    log_level: typing.Union[int, str],
    exc_info=False,
) -> typing.Callable[["RetryCallState"], None]:
    """After call strategy that logs to loguru logger the finished attempt."""

    def log_it(retry_state: "RetryCallState") -> None:
        if retry_state.outcome.failed:
            exception = retry_state.outcome.exception()
            verb, value = "raised", f"{type(exception).__name__}: {exception}"

            if exc_info:
                local_exc_info = get_exc_info_from_future(retry_state.outcome)
            else:
                local_exc_info = False
        else:
            verb, value = "returned", retry_state.outcome.result()
            local_exc_info = False  # exc_info does not apply when no exception

        callback_name = _utils.get_callback_name(retry_state.fn)
        # FIXME: currently fails
        # sleep_time = getattr(retry_state.next_action, "sleep")
        nth_attempt = _utils.to_ordinal(retry_state.attempt_number)
        logger.opt(exception=local_exc_info).log(
            log_level,
            # f"Retrying {callback_name} in {sleep_time:.2f} seconds as its"
            f"Retrying {callback_name} soonish as its"
            f" {nth_attempt} attempt {verb} {value}.",
        )

    return log_it
