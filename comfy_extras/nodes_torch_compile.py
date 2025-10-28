from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.torch_helpers import set_torch_compile_wrapper


class TorchCompileModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TorchCompileModel",
            category="_for_testing",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "backend",
                    options=["inductor", "cudagraphs", "max"],
                ),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, backend) -> io.NodeOutput:
        m = model.clone()
        if backend == "max":
            from torch_max_backend import max_backend, register_max_devices
            register_max_devices()
            backend = max_backend
            
        set_torch_compile_wrapper(model=m, backend=backend)
        return io.NodeOutput(m)


class TorchCompileExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TorchCompileModel,
        ]


async def comfy_entrypoint() -> TorchCompileExtension:
    return TorchCompileExtension()
