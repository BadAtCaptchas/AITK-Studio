def ensure_pkg_resources_packaging():
    try:
        import packaging
        import packaging.version  # noqa: F401 - exposes packaging.version for legacy callers.
        import pkg_resources
    except ImportError:
        return

    if not hasattr(pkg_resources, "packaging"):
        pkg_resources.packaging = packaging
