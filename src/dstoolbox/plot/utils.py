import json
import altair as alt
from urllib import request


def set_altair_locale(locale: str) -> None:
    """Set Altair locale.

    Parameters
    ----------
    locale : str
        name of the locale, e.g. "en-GB"
    """

    format_url = (
        f"https://raw.githubusercontent.com/d3/d3-format/main/locale/{locale}.json"
    )
    with request.urlopen(format_url) as f:
        uk_format = json.load(f)
    time_format_url = f"https://raw.githubusercontent.com/d3/d3-time-format/master/locale/{locale}.json"
    with request.urlopen(time_format_url) as f:
        uk_time_format = json.load(f)
    alt.renderers.set_embed_options(
        formatLocale=uk_format, timeFormatLocale=uk_time_format
    )
