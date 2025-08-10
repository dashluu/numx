from numx.core import Array


class Module:
    def parameters(self) -> list[Array]:
        params: list[Array] = []
        has_submodules = any(isinstance(v, Module) for v in self.__dict__.values())
        if not has_submodules:
            for _, value in self.__dict__.items():
                if isinstance(value, Array):
                    params.append(value)
            return params
        for _, value in self.__dict__.items():
            if isinstance(value, Module):
                params += value.parameters()
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Array:
        raise NotImplementedError()
