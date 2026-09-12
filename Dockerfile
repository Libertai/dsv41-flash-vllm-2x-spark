# DeepSeek-V4.1-Flash serving image for DGX Spark / GB10 (aarch64, sm_121).
#
# vLLM gained DeepseekV41ForCausalLM in https://github.com/vllm-project/vllm/pull/56214.
# Any aarch64 nightly at or after that merge works; this build is pinned to the one
# this recipe was verified against.
#
#   vllm 0.1.1.dev16+gc191787a6 | torch 2.13.0+cu130 | flashinfer 0.6.18.post1 | CUDA 13.0
#
# ⚠️ Install the nightly BY URL. The nightly calls itself "0.1.0", so
#    `pip install --pre vllm` silently prefers PyPI's stable release instead.
# ⚠️ The wheel index hrefs are relative ("../../<sha>/..."), so there is no "/vllm/"
#    path segment, and the "+" in the version must stay percent-encoded.
FROM nvcr.io/nvidia/vllm:nightly-aarch64-cu130

ARG VLLM_WHEEL_URL
RUN test -n "$VLLM_WHEEL_URL" || (echo "pass --build-arg VLLM_WHEEL_URL=<nightly wheel url>" && false) \
 && pip install --no-cache-dir "$VLLM_WHEEL_URL"

# Fail the build now rather than at serve time if the architecture is not registered.
RUN python3 -c "from vllm.models.registry import ModelRegistry as R; \
    assert 'DeepseekV41ForCausalLM' in R.get_supported_archs(), 'DSV4.1 not in this nightly'; \
    import vllm, torch, flashinfer; print(vllm.__version__, torch.__version__, flashinfer.__version__)"
