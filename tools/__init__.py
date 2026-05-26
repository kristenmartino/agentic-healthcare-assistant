"""Tool modules: thin wrappers over external resources / databases.

Submodules are imported lazily — `tools.vector_index` pulls numpy, which we
don't want forced on every test that only touches `tools.ehr`.
"""
