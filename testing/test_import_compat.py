import unittest
import warnings

from toolkit.import_compat import ensure_pkg_resources_packaging


class ImportCompatTest(unittest.TestCase):
    def test_pkg_resources_packaging_alias_is_restored_for_clip(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pkg_resources
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        original = getattr(pkg_resources, "packaging", None)
        had_original = hasattr(pkg_resources, "packaging")
        if had_original:
            delattr(pkg_resources, "packaging")
        try:
            ensure_pkg_resources_packaging()

            from pkg_resources import packaging

            self.assertTrue(hasattr(packaging, "version"))
        finally:
            if had_original:
                pkg_resources.packaging = original
            elif hasattr(pkg_resources, "packaging"):
                delattr(pkg_resources, "packaging")


if __name__ == "__main__":
    unittest.main()
