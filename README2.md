```shell
uv init --python=3.12
```

add

```toml
# TODO torch-backend still experimental, check if there is new solution
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu124", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
```

```shell
uv add -r ./requirements.txt
```