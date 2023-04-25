def spec2title(spec: str) -> str:
    """
    Convert a spec to a snakecased str title

    Parameters
    ----------
    spec: str
        spec to convert

    Returns
    -------
    title: str
        title converted from the given spec
    """

    title: str = spec.replace(" ", "_").replace("&", "and").replace("|", "or")

    return title
