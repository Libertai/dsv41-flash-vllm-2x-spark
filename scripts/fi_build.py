from flashinfer.mla._sparse_mla_sm120 import gen_sparse_mla_sm120_module
s = gen_sparse_mla_sm120_module()
print("aot present (must be False):", s.is_aot, flush=True)
assert not s.is_aot, "AOT .so still visible -- the empty bind-mount did not take"
print("built and loaded:", s.build_and_load(), flush=True)
