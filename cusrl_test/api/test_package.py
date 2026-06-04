import cusrl


def test_import_all_from_top_level_package():
    namespace: dict[str, object] = {}
    exec("from cusrl import *", namespace, namespace)

    for name in cusrl.__all__:
        assert name in namespace
