from comfy_api.torch_helpers import set_torch_compile_wrapper


class TorchCompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "backend": (["inductor", "cudagraphs", "max"],),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"
    EXPERIMENTAL = True

    def patch(self, model, backend):
        m = model.clone()
        if backend == "max":
            from torch_max_backend.compiler import max_backend
            backend = max_backend
            
        set_torch_compile_wrapper(model=m, backend=backend)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "TorchCompileModel": TorchCompileModel,
}
