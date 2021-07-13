import typing

from tenacity import _utils

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
            ex = retry_state.outcome.exception()
            verb, value = "raised", f"{ex.__class__.__name__}: {ex}"

            if exc_info:
                local_exc_info = retry_state.outcome.exception()
            else:
                local_exc_info = False
        else:
            verb, value = "returned", retry_state.outcome.result()
            local_exc_info = False  # exc_info does not apply when no exception

        nth_attempt = _utils.to_ordinal(retry_state.attempt_number)
        logger.opt(exception=local_exc_info).log(
            log_level,
            f"Retrying {_utils.get_callback_name(retry_state.fn)} "
            f"for the {nth_attempt} time "
            f"in {retry_state.next_action.sleep} seconds as it {verb} {value}.",
        )

    return log_it
