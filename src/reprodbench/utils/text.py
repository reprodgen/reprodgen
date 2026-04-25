def preview(text: str | None, n: int = 500) -> str:
    """
    Docstring for preview

    :param text: Description
    :type text: str | None
    :param n: Description
    :type n: int
    :return: Description
    :rtype: str
    """
    if text is None:
        return "<None>"
    t = str(text)
    t = t.replace("\r\n", "\n")
    if len(t) <= n:
        return t
    return t[:n] + "\n...<truncated>..."
