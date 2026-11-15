# mesh-attribute-quantizer

Python 网格属性量化库：位置按包围盒量化、法线八面体编码、UV 按岛范围量化、
权重保归一化；相同原始顶点稳定焊接为同一量化顶点（共享边无裂缝），输出交错
顶点布局、组件偏移、对齐信息与每属性最大误差。纯标准库，无第三方依赖。

## 用法

```python
from mesh_quantizer import AttributeSpec, Mesh, MeshQuantizer, POSITION, NORMAL, UV, WEIGHT

specs = [
    AttributeSpec(POSITION, bits=16, max_error=1e-3),
    AttributeSpec(NORMAL,   bits=10, max_error=5e-2),   # 八面体编码为 2 分量
    AttributeSpec(UV,       bits=12, max_error=1e-3),   # 按 UV 岛分别定范围
    AttributeSpec(WEIGHT,   bits=8,  max_error=1e-2),   # 最大余数法保持和为 1
]
q = MeshQuantizer(specs, alignment=4, strict=True)  # strict: 超预算即抛错
qmesh, ranges = q.quantize(mesh)                    # mesh: Mesh(positions, indices, ...)

qmesh.buffer        # 交错顶点字节流
qmesh.stride        # 顶点步长（按 alignment 补齐）
qmesh.offsets       # 各属性在顶点内的字节偏移
qmesh.indices       # 焊接重映射后的索引
qmesh.max_error     # 各属性解码后的实测最大误差
qmesh.within_budget # 各属性是否满足误差预算
qmesh.welded_away   # 焊接掉的顶点数
qmesh.sanitized     # 各属性被替换的 NaN/inf 分量数
```

## 测试

```
python3 -m unittest test_mesh_quantizer -v
```

覆盖：极窄包围盒（含退化轴）、镜像 UV 岛、零权重、NaN/inf 清洗、共享边焊接
无裂缝、八面体法线回环、交错布局对齐、误差预算校验。
