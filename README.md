# OrdinaryDiffEq.mbt

[MoonBit](https://www.moonbitlang.com) 移植的 [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)：一套**逐行对齐上游**的常微分方程求解器，覆盖显式 Runge–Kutta、Rosenbrock 与变阶变步长 BDF 三大家族，自带稠密输出、自适应步长控制、tstops、saveat 与 STALD 检测。

无任何 Julia 依赖、无 FFI：数值内核（稠密 LU、Newton 迭代、有限差分 Jacobian、数学函数）全部用 MoonBit 重写，可在 **native / js / wasm-gc** 三个目标上编译运行。

- **包名**：`riantr/ordinarydiffeq`
- **模块**：单 package（`src/moon.mod` + `src/src/moon.pkg`，`pkgtype: "library"`）
- **测试**：150 个测试，三目标全绿

## 求解器清单

| 求解器 | 类型 | 阶 | 说明 |
| --- | --- | --- | --- |
| `Euler` | 显式 | 1 | 教学/对照 |
| `Tsit5` | 显式 RK | 5 | Tsitouras 5(4) 嵌入对，通用首选 |
| `Vern7` | 显式 RK | 7 | Verner 7(6)，高精度非刚性 |
| `Vern9` | 显式 RK | 9 | Verner 9(8)，最高精度非刚性 |
| `ExplicitRK` | 显式 RK | 由 tableau 决定 | 通用 tableau 驱动（Heun / RK4 / BS3 / RKF4 / Dormand–Prince） |
| `Rosenbrock23` | 半隐式 | 2 | 低阶刚性，W-方法 |
| `Rodas5P` | 半隐式 | 5 | 高阶刚性，ROS 系 |
| `FBDF` | 多步 BDF | 1–5 变阶 | 固定修正 BDF + Nordsieck 历史 + Newton 迭代 |

所有方法都支持自适应与固定步长两种模式、稠密输出与回调式的 saveat/tstops 截断。

## 快速上手

```moonbit
// u' = -u, u(0) = 1，精确解 exp(-t)
let prob = @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(_t, u, du) {
  du.unsafe_set(0, -u.unsafe_get(0))
})

// 自适应：Tsit5 + 默认 PI 控制器
let sol = @ordinarydiffeq.solve(prob, @ordinarydiffeq.Tsit5::new(),
  @ordinarydiffeq.SolveOptions::new())

// 固定步长 dt = 0.01
let sol2 = @ordinarydiffeq.solve(prob, @ordinarydiffeq.Tsit5::new(),
  @ordinarydiffeq.SolveOptions::with_dt(@ordinarydiffeq.SolveOptions::new(), 0.01))

// 端点值
let u1 = sol.u.unsafe_get(sol.u.length() - 1).unsafe_get(0)
// 任意时刻取值（需要 dense = true，或 saveat 命中）
let umid = sol.call(0.5)
```

`OdeProblem` 区分 in-place（`f(t, u, du)` 写 `du`）与 out-of-place（`f(t, u) -> u`），`SolveOptions` 提供一组链式构造器：

```moonbit
SolveOptions::with_dt(opts, dt)          // 固定步长
SolveOptions::with_adaptive(opts, true)  // 自适应（默认 true）
SolveOptions::with_reltol(opts, 1e-8)
SolveOptions::with_abstol(opts, 1e-10)
SolveOptions::with_dtmax / with_dtmin
SolveOptions::with_dense(opts, true)     // 启用求解器专属插值
SolveOptions::with_saveat(opts, ts)      // 只在 ts 上保存
SolveOptions::with_tstops(opts, ts)      // 步长强制落在 ts 上
SolveOptions::with_maxiters(opts, n)
```

返回码 `sol.retcode`：`Success` / `MaxIters` / `DtMinUnstable` / `Unstable` / `InitialFailure`。

### 刚性例子

```moonbit
// u' = -1000(u - cos(t))，强刚性，显式方法需要极小步长
let prob = @ordinarydiffeq.OdeProblem::iip(0.0, 1.0, [1.0], fn(t, u, du) {
  du.unsafe_set(0, -1000.0 * (u.unsafe_get(0) - @math.cos(t)))
})
let sol = @ordinarydiffeq.solve(prob, @ordinarydiffeq.Rodas5P::new(),
  @ordinarydiffeq.SolveOptions::new())
```

## 架构

```
src/src/
├── problem.mbt     OdeProblem / SolveOptions（含 IIF 与 OOP 两种函数签名）
├── solver.mbt      Solver trait：order / is_fsal / stage_count / perform_step /
│                   interpolant / fill_dense_stages / uses_bdf_controllers
├── solve.mbt       主循环：loopheader / loopfooter / 步接受与拒绝 / saveat / tstops
├── integrator.mbt  Integrator 状态 + FBDFCache（Nordsieck 历史、Newton 缓存）
├── control.mbt     PI 步长控制器（Gustafsson）+ 初始步长启发式
├── numcore.mbt     加权 RMS 范数、fastpower 等数值原语
├── mathshim.mbt    exp / sqrt / log / 幂等数学函数
├── linalg_lu.mbt   自写稠密 LU（部分选主元）+ 线性求解
├── jacobian.mbt    有限差分 Jacobian、J→W 变换、w_solve
├── stald.mbt       STALD（刚性检测）
├── bdf_utils.mbt   BDF 系数表、Chebyshev 重采样、Lagrange 插值、Fornberg 权重
├── bdf_interp.mbt  FBDF 的 Nordsieck 插值
├── alg_*.mbt       各求解器（每族一个文件，系数表独立成 *_coeffs.mbt）
└── *_test.mbt      黑盒端到端测试（经 `@ordinarydiffeq.` 前缀引用包接口）
```

### 移植保真度

本项目的目标是**行为可对上**，不是性能最优。每个求解器文件都带有上游源码定位注释（如 `bdf_perform_step.jl:1491`），关键分支按上游逐行转录，包括几处“看似可疑实则忠实”的细节：

- `do_newJW` 的 fresh-Jacobian 分支落到公共 `wbad` 计算，而非直接返回 `(false, true)`；
- `choose_order!` 降阶分支里 Fornberg 权重表第 0 列被清零，导致 `u` 项恒为 0（只影响启发式阶梯，不影响步长公式）；
- FBDF 的 `fsallast` 走代数恒等式（`α/(γdt)·u - nlsolver.tmp`），不额外调用 `f`、不计 `nf`；
- `NLStatus::TryAgain` 是 `SlowConvergence` 的别名（值 0），`nlsolvefail` 判据是 `status ≤ 0`；
- 固定步长 FBDF 等价于隐式 Euler（`u_{n+1} = u_n/(1+dt)`）。

与上游的已知差异（均为有意取舍）：

1. **无注册表/缓存类型分发**：上游用 `alg → cache → controller` 的类型链，本移植用 `Solver` trait 加一个 `uses_bdf_controllers` 标志，由 `solve()` 一次性解析到 `integ.bdf_active`。
2. **`NLSolver` + `NLNewtonCache` 合并进 `FBDFCache`**：上游两个结构体的字段原样保留，只是共处一地。
3. **`derivative_discontinuity` 标志由 `restart_pending` 承担**，置位/读取/清除窗口与上游一致。
4. **无 mass matrix、无 callback 系统、无 事件检测**：超出 M5 范围，`OdeFunction` 只承载 `f`。
5. 误差范数对 `Double` 恒为 RMS 范数（上游 `norm` 在复数/自定义数上有别的选择）。

## 开发

```bash
cd src
moon test --target native   # 或 js / wasm-gc
moon check                 # 类型检查
moon fmt                   # 格式化
moon info                  # 重新生成 pkg.generated.mbti
```

测试全部内嵌在源码里（`test` 块与实现同文件，便于引用内部函数；黑盒测试集中在 `*_test.mbt`，经 `@ordinarydiffeq.` 前缀只访问公开接口）。`tools/` 下是移植过程中用到的 Python 参考实现（Fornberg 权重、terk 估计、Newton 内核等），仅作交叉验证，不参与构建。

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。
