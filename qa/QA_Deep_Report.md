# SciML 工作区 · 深度 QA 审计报告

> 审计对象：`D:/src/MiniMax/Projects/SciML`（MoonBit 移植版 OrdinaryDiffEq + Julia 参考实现 + 探针/框架子项目）
> 审计方式：**实证驱动**（实际运行 `moon check` / `moon test --enable-coverage` / 静态扫描），非纯文档评审
> 审计日期：2026-10-25 ｜ 工具链：MoonBit `0.1.20261023`、Python 3.13

---

## 0. 执行摘要

| 项 | 结果 |
|---|---|
| 类型检查 (`moon check`) | ✅ 通过（0 错误） |
| 单元测试（native 目标） | ✅ **150 个断言 / 91 个 `test` 块，全部通过**，运行 ~1s |
| 测试矩阵 | ✅ CI 已覆盖 `native / js / wasm / wasm-gc` 四目标 |
| 代码覆盖率（native） | 🟢 **91.0% 行块覆盖**（22/22 含可执行代码的模块均被覆盖；弱点在 `bdf_utils` 56%、`dense` 66%）｜CI 已加门禁（≥90% 整体 / ≥80% 单模块）+ HTML 上报 |
| 显式错误处理 | 🟢 已有基于 `Retcode` 的返回码体系（`Success/MaxIters/DtMinUnstable/Unstable/InitialFailure`）：`check_error`（`solve.mbt:75`）每迭代检查并在非 Success 立即返回；`lu_factor`（`linalg_lu.mbt:43`）对奇异/NaN 主元返回 `false` 转步失败；全代码**无 `panic`/`abort`/`raise` 关键字**（错误以返回码表达，符合 SciML `sol.retcode` 约定） |
| 安全敏感代码 | 🟡 `unsafe_get/unsafe_set` 出现 **1051 次**（数值内核普遍未做边界检查）；端口内 `extern`=0、FFI=无 |
| 依赖与供应链 | 🟢 仅 `depprobe2` 引入外部包 `riantr/moonbit_image@0.3.7`（**已精确钉 `@0.3.7`**）；MoonBit 无 `moon-lock.json`（版本直接钉在 `moon.mod`）；CI 加 `deps_check.py` 钉版本守卫；Python 工具仅用标准库 |
| 性能/基准 | 🟢 有 `bench_test.mbt`（用 `core/bench`）；**端口 CI 已加 `benchmark` 作业 + 50ms 合理性预算门**，JSON 结果上传 artifact（精确回归阈值待建基线；仅 Julia 侧有 `benchmark.yml`） |
| Fuzz / Gherkin / 负载 | 🟡 **Fuzz/属性测试已落地**（`src/src/fuzz_test.mbt`，7 个随机化测试，随四目标 `test` 矩阵运行）；Gherkin 数值库低优先、有意不做；负载不适用 |
| 重复代码 | 🟡 结构性重复为主：`alg_vern7`/`alg_vern9` 共享 **109** 个相同 5 行片段；求解器脚手架跨 8 文件重复；测试样板跨 10 文件重复 |
| 指标门禁 | 🟢 **CI 已加 `metrics` 作业 + `qa/metrics_gate.py`**：单文件决策点 ≤120、`unsafe_get/set` ≤1600、重复率 ≤20%、`test` 块 ≥90，`extern`/`panic`/`abort`/`raise` 恒为 0（详见 §6.6） |

**总体判断**：端口的**功能正确性质量很高**（四目标全绿、覆盖率 91%、数值契约对齐上游），且**工程化 QA 门禁已相当完整**——测试矩阵、类型检查、覆盖率门禁+上报、依赖钉版本守卫、基准 CI、Fuzz/属性测试、质量指标门禁均已落地；错误以 `Retcode` 返回码表达（无裸 panic）。剩余弱项集中在**精确性能回归阈值（需基线）** 与**跨实现（vs Julia）集成比对显式化**。下面的 16 维度逐项给出证据与修复路线。

---

## 1. 评分卡（16 维度）

等级：🟢 优秀 ｜ 🟡 良好/部分 ｜ 🔴 缺失/风险 ｜ ⚪ 不适用(N/A)

| # | 维度 | 等级 | 一句话结论 |
|---|------|------|-----------|
| 1 | SAST（静态安全） | 🟡 | 无 `unsafe`/`extern`/`panic` 关键字，但 1051 处 `unsafe_get/set` 未做边界校验；CI 无 SAST 步骤 |
| 2 | DAST（动态安全） | ⚪ | 纯数值库无网络/外部输入面，常规 DAST 不适用；可用 sanitizer/模糊运行时替代 |
| 3 | 依赖（Dependency） | 🟢 | 仅 1 个外部 MoonBit 包且已精确钉 `@0.3.7`；MoonBit 用 `@version` 钉版本（无 `moon-lock.json`），CI `deps_check.py` 守卫防漂移；Python 工具零三方依赖 |
| 4 | 单元（Unit） | 🟢 | 91 test 块 / 150 断言，四目标全绿，模块化清晰 |
| 5 | 集成（Integration） | 🟡 | 求解器↔积分器↔稠密↔解 已在测试内贯通；缺与 Julia 上游的跨实现集成比对 |
| 6 | API（接口） | 🟡 | 公共 API 由 solver/options 测试覆盖；缺独立的 API 契约/示例快照测试 |
| 7 | 契约（Contract） | 🟢 | `tools/*_expected.py` 生成上游“期望解”，测试逐值比对 → 隐式数值契约测试（亮点） |
| 8 | 组件（Component） | 🟢 | 每个算法族独立 `*.mbt` + 独立 `*_test.mbt`，组件隔离良好 |
| 9 | Gherkin（BDD） | 🔴 | 全工作区无 `.feature`；数值库低优先，可不做 |
| 10 | E2E（端到端） | 🟡 | 四目标（native/js/wasm/wasm-gc）运行即多平台 E2E；无“应用级”E2E（`mbgui` 有 smoke_test） |
| 11 | 覆盖率（Coverage） | 🟢 | 91% 行块覆盖；**CI 已加门禁（≥90% 整体 / ≥80% 单模块，bdf_utils/dense 豁免）+ HTML 上报 artifact**；`bdf_utils`/`dense` 偏低 |
| 12 | Fuzz（模糊） | 🟢 | `src/src/fuzz_test.mbt` 已加 7 个确定性随机化测试（xorshift64 PRNG + 属性/模糊）：线性衰减精度、常数 RHS 精确性、谐振子能量守恒、Tsit5↔Vern9 一致性、固定步确定性、随机多项式 RHS 不崩溃、刚性显式求解干净失败；随四目标 `moon test` 矩阵运行 |
| 13 | 性能（Performance） | 🟢 | 有 `bench_test`；**CI 已加 `benchmark` 作业（native/release）+ 50ms 合理性预算门**，JSON 结果上传 artifact 供趋势追踪；精确回归阈值待建基线 |
| 14 | 负载（Load） | ⚪ | 非服务类库，无并发/吞吐面；大规模 N 的内存/稳定性未测（低优先） |
| 15 | 重复代码（Duplication） | 🟡 | 多为代码生成家族的结构性重复（可接受），但测试样板与求解器脚手架可抽公共helper |
| 16 | 指标（Metrics） | 🟢 | CI 已加 `metrics` 作业：单文件决策点 ≤120、unsafe ≤1600、重复率 ≤20%、测试块 ≥90、extern/panic/abort/raise 恒 0；并修正了原扫描把注释计入导致的复杂度高估 |

---

## 2. 实证数据

### 2.1 覆盖率（native 目标，合并 blackbox + internal 测试包）

| 模块 | 行块覆盖 | 覆盖块/总块 | 模块覆盖 |
|------|---------|------------|---------|
| bdf_utils | **55.9%** | 128/229 | 100% |
| dense | **65.9%** | 29/44 | 100% |
| control | 85.4% | 88/103 | 100% |
| mathshim | 86.4% | 19/22 | 100% |
| solution | 88.6% | 101/114 | 100% |
| solve | 92.0% | 195/212 | 100% |
| problem | 92.6% | 25/27 | 100% |
| alg_vern9 | 93.5% | 343/367 | 100% |
| stald | 93.8% | 210/224 | 100% |
| alg_fbdf | 94.7% | 266/281 | 100% |
| jacobian | 95.2% | 59/62 | 100% |
| alg_rosenbrock23 | 96.0% | 95/99 | 100% |
| linalg_lu | 98.1% | 51/52 | 100% |
| alg_rodas5p | 99.1% | 114/115 | 100% |
| alg_vern7 | 99.4% | 178/179 | 100% |
| alg_explicit_rk / tsit5 / bdf_interp / integrator / numcore / solver / tableaus | 100% | — | 100% |
| *_coeffs（纯数据） | n/a | 0 块 | — |

> 说明：`*_coeffs` 为纯系数常量表，无可插桩语句，按 0 块计，不计入缺口。
> **总计：2546 行块，覆盖 2317（91.0%）；22 个含可执行代码的模块全部被覆盖（100% 模块覆盖）。** 注：此前“88% 函数覆盖”是把 3 个纯系数数据模块（0 可插桩块）误计为“未覆盖函数”；MoonBit 插桩为模块级，不暴露函数级粒度，故统一以模块覆盖报告（右侧“模块覆盖”列=100% 表示该模块有 ≥1 个被覆盖块）。

### 2.2 测试清单（端口 `src/src/`）

| 测试文件 | test 块数 | 代码行 |
|---------|---------|-------|
| adaptive_test.mbt | 13 | 385 |
| explicit_rk_test.mbt | 17 | 472 |
| fbdf_test.mbt | 10 | 309 |
| rodas5p_test.mbt | 10 | 354 |
| rosenbrock23_test.mbt | 11 | 318 |
| solve_test.mbt | 9 | 225 |
| tsit5_test.mbt | 5 | 156 |
| vern7_test.mbt | 8 | 169 |
| vern9_test.mbt | 8 | 172 |
| bench_test.mbt（基准，非断言测试） | 0 | 152 |
| **合计** | **91** | ~2612 |

> README 标注“150 个测试”≈ 91 个 `test` 块内的 `@test` 断言总数。

### 2.3 SAST 信号扫描（端口源码，非测试）

| 信号 | 计数 | 落点 |
|------|------|------|
| `unsafe`（关键字块） | 0 | — |
| `unsafe_get` / `unsafe_set`（方法） | **1051** | 20 个源文件全覆盖（数值内核） |
| `extern` | 0 | 端口无 FFI（✅ 与 README“无 FFI”一致） |
| `panic` / `abort` | 0 | — |
| `raise` / `error`(显式) | 0 | 初版 grep 曾报 1 处，系 `control.mbt:114` 文档注释里的 “error (” 字样（误报）；现行扫描已先剥离注释，故真实代码 0 处 |
| `unwrap` / `expect` | 0 | — |
| TODO / FIXME / HACK | 0 | — |

> 关联子项目：`mbgui` 含 **244** 处 `extern`（GUI/跨端 FFI，预期内，属其 SAST 边界）；`depprobe2`/`ffiprobe` 无 `extern`。

### 2.4 复杂度热点（决策点代理 = if/match/while/for/loop/guard/try/else 计数）

> **口径更新（2026-10-25）**：计数改为**剥离 `//` 注释后**再统计。此前直接对原文计数，会把文档注释里的 `if`/`while`/`else` 等词一并计入，系统性**高估**复杂度。故下表数值低于本报告初版；以新版为准。

| 文件 | 代码行 | 决策点 | 注释率 | 最长行 |
|------|-------|-------|-------|-------|
| alg_fbdf.mbt | 874 | **102** | 36% | 88 |
| solve.mbt | 399 | **91** | 32% | 86 |
| stald.mbt | 527 | **67** | 23% | 83 |
| bdf_utils.mbt | 592 | **55** | 36% | 94 |
| alg_explicit_rk.mbt | 224 | **29** | 40% | 83 |
| alg_rodas5p.mbt | 301 | **28** | 39% | 99 |
| control.mbt | 267 | **25** | 34% | 80 |
| alg_vern9.mbt | 1718 | **23** | 13%（多为系数表） | **182** |
| jacobian.mbt | 405 | **17** | 33% | 79 |

### 2.5 重复代码（5 行滑动窗口哈希，仅列 Top）

| 重复簇 | 出现次数 | 涉及文件 |
|-------|---------|---------|
| Verner 内核片段 | **109** | alg_vern7 ↔ alg_vern9 |
| Verner 内核片段（变体） | 31 / 30 / 28 | alg_vern7 ↔ alg_vern9 |
| 通用求解器脚手架 | 15 | rodas5p/rosenbrock23/tsit5/vern7/vern9/numcore |
| 通用求解器脚手架 | 13 | + jacobian |
| 测试样板（建 problem→solve→断言） | 10 | 全部 9 个测试文件 |
| `perform_step` 模板 | 8×5 | 8 个算法/求解文件 |

> 判读：Verner 7/9 同源代码生成、求解器脚手架与测试样板属**结构性重复**，多数可接受；但 `vern7/vern9` 的 109 处一致片段与测试样板 10× 重复具备抽取为共享模板/helper 的净收益。

---

## 3. 分维度深度发现与建议

### 1) SAST 🟡
- **发现**：端口无 `unsafe` 关键字块、无 `extern`、无显式 `panic/abort/raise`，类型系统强；但数值内核密集使用 `unsafe_get/unsafe_set`（1051 处），在索引已被逻辑保证在界内时高效，一旦上游改动或边界条件遗漏即触发**未定义行为/越界 panic**。
- **建议**：
  1. CI 增加 SAST 步骤：正则/`moon check` 告警 + 一个轻量扫描脚本（见 `qa/qa_analysis.py` 的 SAST 段）将 `unsafe_get/set` 用量纳入趋势监控。
  2. 对索引来源（循环上界、`u.length()` 派生）做注释/断言闭环；热点路径（LU 分解、Jacobian 有限差分）加 `debug_assert` 边界。
  3. 评估 `moonbitlang/core` 是否已提供带边界检查的 `get`/`set` 供非热点路径替换。

### 2) DAST ⚪
- **发现**：纯数值库无网络/文件/进程外部输入面，常规 DAST（注入、SSRF 等）不适用。
- **建议**：以**运行时 sanitizer + 模糊**替代（见 Fuzz）。对 `ffiprobe`（FFI 探针）与 `mbgui`（244 extern）这类有边界交互的子项目，可对其 JS/native 后端做输入合法性 DAST。

### 3) 依赖 🟡
- **发现**：端口仅依赖 `moonbitlang/core/{bench,double,math}`（标准库，可信）；`depprobe2` 引入外部 `riantr/moonbit_image@0.3.7`（**已精确钉 `@0.3.7`**）。**MoonBit 没有 `moon-lock.json`**——依赖版本直接钉在 `moon.mod` 的 `import` 行（`@x.y.z`），这是 MoonBit 原生的可复现机制；Python 代码生成工具仅用标准库（优秀）。
- **建议**：
  1. ~~提交 `moon-lock.json`~~ → MoonBit 原生以 `@version` 钉版本（无 lockfile 概念）；已加 `qa/deps_check.py` CI 守卫，强制所有外部依赖带精确 `x.y.z` 钉，杜绝静默漂移（见 §5 附录）。
  2. 对 `riantr/moonbit_image` 持续关注其发布/漏洞通告；考虑 Dependabot（Julia 侧已用，可借鉴）。
  3. 为 Python 工具加 `requirements` 冻结（当前 stdlib-only，风险低，但工具脚本未来可能引入 `numpy`/`sympy` 等）。

### 4) 单元 🟢
- **发现**：91 test 块 / 150 断言，native 全绿、四目标矩阵覆盖，组件级测试文件齐全。
- **建议**：保持；将“四目标全绿”作为合并门禁（已部分实现），并为每个新增求解器强制配套 `*_test.mbt`。

### 5) 集成 🟡
- **发现**：端口内求解器↔积分器↔稠密输出↔解对象已在各测试贯通；但与 **Julia 上游的跨实现集成比对**无自动化（仅人工/代码生成期比对）。
- **建议**：新增“对拍”集成测试：用同一组 IVP（含刚性/病态/变步长边界）在端口与 `tools/*_expected.py` 的上游数值间做容差比对，纳入 CI（当前契约测试已部分覆盖，建议显式化为集成层）。

### 6) API 🟡
- **发现**：`solve`/`SolveOptions`/`OdeProblem` 由 solver/options 测试覆盖，但缺**独立 API 契约/示例快照**测试，README 示例易随重构失同步。
- **建议**：将 README 中的 Quick Start 片段提升为 `doc-test`（MoonBit 支持文档测试），保证示例永不过期；对公共函数签名变更加 `moon info` 差异检查。

### 7) 契约 🟢（亮点）
- **发现**：`tools/{m2,m3,m4,fbdf}_expected.py` 与 `extract_*.py` 从上游抽取“期望解/系数”，测试逐值比对——这是**隐式但扎实的数值契约测试**，是端口正确性的核心保障。
- **建议**：将其显式化：① 在 CI 标注“契约测试”阶段；② 扩大对拍的算法/问题覆盖（当前 Rosenbrock/BDF/Verner 有，ExplicitRK/TSIT5 有，建议补齐边界情形：奇异 Jacobian、dtmin、事件/不连续）。

### 8) 组件 🟢
- **发现**：每个算法族独立模块 + 独立测试，组件隔离良好，利于并行维护与替换。
- **建议**：维持；组件边界用 `moon.pkg` 的 `pub` 可见性严格控制（当前 `warnings` 已启用 `missing_priv`，良好）。

### 9) Gherkin 🔴（低优先）
- **发现**：无 `.feature` / BDD。
- **建议**：对数值库**不建议**强行引入 Gherkin；若需向非工程干系人展示行为，可用 Markdown 场景表替代。标记“有意不做”。

### 10) E2E 🟡
- **发现**：四目标（native/js/wasm/wasm-gc）运行即多平台 E2E；`mbgui` 有 `smoke_test.mjs`（浏览器端冒烟）。
- **建议**：端口增加“全求解器 × 全目标”的最小可运行 E2E 清单（已有测试覆盖，建议加一个汇总清单防止目标退化）；`mbgui` 的 smoke 测试纳入 CI 门禁。

### 11) 覆盖率 🟡
- **发现**：native 91% 行块覆盖；`bdf_utils`(56%)、`dense`(66%)、`control`(85%)、`mathshim`(86%) 偏低。覆盖率门禁与上报**已于 2026-10-25 落地**（见 §5 附录）：CI `coverage` 作业跑插桩测试 + `qa/coverage_gate.py` 门禁（≥90% 整体 / ≥80% 单模块，bdf_utils/dense 豁免至 45% 硬底）+ HTML 上报 artifact + PR 摘要。
- **建议**：
  1. ~~CI 增加插桩测试 + 阈值门禁~~ → 已完成（`qa/coverage_gate.py`，整体 ≥90% / 单模块 ≥80%，bdf_utils/dense 豁免）。
  2. 针对 `bdf_utils`/`dense` 补分支测试：变阶切换、插值边界、步长拒绝路径（提升这两处覆盖率，争取移除豁免）。
  3. ~~将覆盖率报告上传~~ → 已完成（GitHub Actions `coverage-report` artifact + `GITHUB_STEP_SUMMARY` PR 摘要）。

### 12) Fuzz 🟢（已落地）
- **发现（原 🔴）**：无任何模糊/属性测试。本会话已补齐——新增 `src/src/fuzz_test.mbt`，用**确定性 xorshift64 PRNG**（固定种子）生成随机输入，共 **7 个随机化 `test` 块**，覆盖约 1160 次随机求解。
- **已实现的属性 / 模糊**：
  1. 标量线性衰减 `u' = -a u`：自适应解须有限、时间单调，并满足容差契约（混合绝对+相对界，避免解衰减到 ~1e-22 时纯相对误差发散）。
  2. 常数 RHS `u' = c`：仿射解析解，端点误差 < 1e-9。
  3. 谐振子 `u' = Ju`（旋转）：能量 ∥u∥² 守恒，相对漂移 < 1e-2。
  4. 光滑多项式 RHS `u' = a·t² − b·t + c`：Tsit5 与 Vern9 端点解一致 < 1e-7（跨方法交叉校验，捕获静默错误）。
  5. 固定步可复现性：同一问题两次求解位级一致（防隐藏全局/种子状态）。
  6. 随机有界多项式 RHS（`u' = c0 + c1·u + c2·u²`，|ci|≤0.5）：**永不崩溃**，返回已知返回码、时间单调；若 `Success` 则状态有限且落点 = `tf`。
  7. 刚性显式求解（`u' = -k·u`, k 至 1e7）：须“干净失败”（`DtMinUnstable`/`Unstable`），**绝不崩溃 / 挂起 / 返回未知状态**。
- **工程化**：套件随四目标 `moon test` 矩阵运行（无独立 CI 作业），门禁天然覆盖；PRNG 固定种子 → 结果可复现、CI 无 flakiness。
- **残留（低优先）**：无 coverage-guided fuzzing（MoonBit 无 libFuzzer 前端，此为随机化测试的等价替代）；奇异矩阵与 NaN 仍归到 `Unstable` 兜底码，未细分诊断。

### 13) 性能 🟢（CI 作业已加，精确回归待基线）
- **发现**：有 `bench_test.mbt`（基于 `core/bench`，8 个基准：TSIT5/Vern7/Rosenbrock23/Rodas5P/FBDF × decay/van der Pol/stiff）。**端口 CI 已于 2026-10-25 加 `benchmark` 作业**（native/release 跑 `moon bench` + `qa/bench_report.py` 解析/汇总），并设 **50ms/基准的“合理性预算”硬门**（捕获灾难性回归/近死循环，而非精密回归阈值——绝对耗时随 runner 波动，紧阈值会误报）；结果 JSON 作为 artifact 上传，支持跨运行趋势比对。Julia 侧有成熟 `benchmark.yml` + `DiffEqDevTools`。
- **建议**：
  1. ~~端口 CI 增加 `benchmark` job~~ → 已完成（`qa/bench_report.py`）。
  2. （可选）建立基准基线：将 artifact 中某次“黄金”运行的 JSON 存为 `qa/bench_baseline.json`，脚本加 `--baseline` 比对模式（回归 > N 倍则告警/失败），实现精密回归门——但需约定基线 runner 以消除机器差异。
  3. 关注四目标间的性能差异（wasm/wasm-gc 的热路径与 native 不同）；可加 wasm-gc 基准作业横向对比。

### 14) 负载 ⚪（低优先）
- **发现**：非服务类库，无并发/吞吐面；大规模 N（大状态向量）下的内存与稳定性未系统测试。
- **建议**：若需，可加“大 N 系统”（如 N≥1000 的线性/刚性系统）内存与耗时压测，作为性能维度的延伸；当前标“不适用/低优先”。

### 15) 重复代码 🟡
- **发现**：结构性重复为主。最显著：`alg_vern7`/`alg_vern9` 共享 **109** 个相同 5 行片段（同源代码生成），求解器脚手架跨 8 文件重复，测试样板跨 10 文件重复。
- **建议**：
  1. `vern7/vern9` 抽取共享 Verner 内核模板（或保留生成脚本并加生成注释，明确“生成即重复”）。
  2. 测试样板抽为 `testkit` helper（`make_problem`/`assert_close`/`run_solver`），降低样板重复、统一容差。
  3. ~~将重复率纳入指标门禁~~ → 已完成：`qa/metrics_gate.py` 对重复率设 ≤20% 门禁（当前 10.9%，见 §6.6）。

### 16) 指标 🟢（门禁已加）
- **发现（原 🔴）**：此前无自动复杂度/重复率/质量门禁，仅人工一次性测算。本会话已补齐——新增 `qa/metrics_gate.py` + CI `metrics` 作业，把质量指标变成**持续执行的门禁**（详见 §6.6）。
- **已门禁的指标**（默认值按基线留足余量，只拦“阶跃式恶化”）：
  - 单文件决策点 ≤ 120（当前最高 102，`alg_fbdf.mbt`）— 拦新增巨型模块。
  - `unsafe_get/unsafe_set` 调用点 ≤ 1600（当前 1051）— 拦越界访问失控式增长。
  - 重复率 ≤ 20.0%（当前 **10.9%**，1737/15981 处 5 行窗口）— 拦复制粘贴蔓延。
  - `test` 块数 ≥ 90（当前 98）— 拦测试被删。
  - **架构不变量恒为 0**：`extern`、`panic`、`abort`、`raise/error`——端口刻意“零 FFI、零 panic（错误一律走 `Retcode`）”，一旦破例立即失败。
- **同时修正的口径**：`qa/qa_analysis.py` 现已在统计前**剥离 `//` 注释**。原实现把注释里的关键词计入，既高估复杂度（初版称 `alg_fbdf` 134 / `solve` 106 / `bdf_utils` 89，实为 **102 / 91 / 55**），也让 `raise/error` 误报 1 处。见 §2.4 口径说明。
- **残留建议**：
  1. 对复杂度热点（`alg_fbdf` 102 / `solve` 91 / `stald` 67 决策点）做评审/拆分，降低认知负荷与回归风险。
  2. 统一格式与行宽（`alg_vern9` 最长行 182，多为系数，可接受；其余应 ≤120）。
  3. 如需趋势视图，可将 `metrics.json` artifact 跨运行留存比对（同类于 `bench_results.json`）。

---

## 4. 优先级修复路线

| 优先级 | 行动 | 维度 | 预期收益 |
|-------|------|------|---------|
| P0 | ~~提交 `moon-lock.json`~~ → MoonBit 无 lockfile；改用 `deps_check.py` 精确版本钉守卫 + 锁定 `riantr/moonbit_image@0.3.7` | 依赖 | 可复现性、供应链稳定（**已完成**） |
| P0 | CI 加覆盖率门禁（≥90% 整体 / ≥80% 单模块，bdf_utils/dense 豁免）+ HTML 上报 | 覆盖率 | 防止回归、可视化趋势（**已完成**） |
| P1 | ~~引入 Fuzz/属性测试~~ → 已完成（`src/src/fuzz_test.mbt`，7 个随机化测试，随四目标矩阵） | Fuzz/健壮性 | 捕获静默错误与崩溃（**已完成**） |
| P1 | ~~显式错误健壮层~~ → 已有 `Retcode` 返回码体系（`solve.mbt` 每迭代 `check_error`、`lu_factor` 奇异/NaN 主元返回 `false` 转步失败），错误以返回码而非裸 panic 表达（符合 SciML `sol.retcode` 约定） | SAST/运行时 | 可观测、可恢复（**已完成**；仅奇异/NaN 细分诊断待补） |
| P1 | 端口基准 CI（合理性预算门）+ JSON 趋势 artifact | 性能 | 防止灾难性性能退化（**已完成**；精密回归阈值待基线） |
| P2 | 测试样板/Verner 内核抽公共 helper | 重复代码 | 可维护性 |
| P2 | ~~质量指标门禁（复杂度/重复率/SAST 趋势）~~ → 已完成：`qa/metrics_gate.py` + CI `metrics` 作业（单文件决策点 ≤120、unsafe ≤1600、重复率 ≤20%、测试块 ≥90、extern/panic/abort/raise 恒为 0） | 指标 | 持续质量门禁、拦阶跃式恶化（**已完成**） |
| P3 | 跨实现（vs Julia）对拍集成测试显式化 | 集成/契约 | 上游同步保障 |
| P3 | README 示例提升为 doc-test | API | 文档永不过期 |

---

## 5. 复现方法

```bash
# 类型检查
cd OrdinaryDiffEq.mbt/src && moon check

# 四目标测试（CI 同款）
for t in native js wasm wasm-gc; do moon test --target $t; done

# 覆盖率（合并 blackbox + internal 测试包）
moon test --target native --enable-coverage
# 原始数据见 _build/moonbit_coverage_*.txt

# 复用本次 QA 测算（覆盖率/测试/SAST/指标/重复代码）
python qa/qa_analysis.py        # 输出结构化 JSON
```

> 注：本次扫描脚本 `qa/qa_analysis.py` 使用绝对路径读取端口源码与覆盖率产物，可直接重跑；如需纳入 CI，建议改为相对路径入参。

---

## 6. 附录：2026-10-25 已落地的 CI 增强

针对 §4 P0 两项、P1 性能项、P1 Fuzz 项与 P1 错误健壮层，已实施并随端口提交：

### 6.1 依赖钉版本守卫（`qa/deps_check.py` + `dependency-pin` 作业）
- **原理**：MoonBit **无 `moon-lock.json`**；依赖版本直接钉在 `moon.mod` 的 `import` 行（`"riantr/moonbit_image@0.3.7"`）。守卫扫描所有 `moon.mod`，要求每个**外部**注册表依赖带精确 `x.y.z` 钉；`moonbitlang/core/*` 工具链内置包豁免。
- **失败即阻断 CI**：任何外部依赖缺 `@` 钉或非精确 semver → 作业非零退出。
- **本地复现**：`python3 qa/deps_check.py --root .`

### 6.2 覆盖率门禁 + 上报（`qa/coverage_gate.py` + `coverage` 作业）
- **流程**：`moon test --target native --enable-coverage` → 合并 `moonbit_coverage_*.txt`（按基础块取 max 命中）→ 计算整体/分模块行块覆盖 → 写 `GITHUB_STEP_SUMMARY`（PR 可见）+ 生成 HTML 报告并作为 `coverage-report` artifact 上传。
- **门禁阈值**：整体 ≥ **90%**；单模块 ≥ **80%**；`bdf_utils`/`dense` 因已知缺口**豁免**（但仍受 **45%** 硬底约束，跌破即失败）。
- **当前实测**：整体 **91.0%** 通过；`bdf_utils` 55.9%、`dense` 65.9% 落在豁免区。
- **本地复现**：`python3 qa/coverage_gate.py --root src --report-html qa/coverage_report.html`
- **退出码语义**：0=通过；1=门禁违反；2=未找到覆盖率产物。

### 6.3 基准 CI（`qa/bench_report.py` + `benchmark` 作业）
- **流程**：`moon bench --target native --release` → `qa/bench_report.py` 解析每条基准的均值耗时与单位（归一化为 ns）→ 写 `GITHUB_STEP_SUMMARY`（PR 可见）+ 生成 JSON 结果并作为 `bench-results` artifact 上传。
- **合理性预算门禁**：每条基准均值 > **50ms** 即失败（`--budget-ms` 可调）。该门仅防灾难性回归/近死循环——绝对耗时随 runner 波动，**不是**精密回归阈值；精密回归需先建基线（见 §13 建议）。
- **当前实测**：8 条基准全过，耗时区间约 **356ns（fbdf stiff）~ 1.27µs（rodas5p stiff）**，全部远低于 50ms 预算。
- **本地复现**：`python3 qa/bench_report.py --bench-dir src/src --json-out qa/bench_results.json`
- **退出码语义**：0=通过；1=某基准突破预算；2=未解析到基准 / `moon bench` 非零退出。

### 6.4 Fuzz / 属性测试套件（`src/src/fuzz_test.mbt`）
- **动机**：补 §4 P1 Fuzz 维度（原 🔴）。MoonBit 无 libFuzzer 前端，故采用**确定性随机化测试**等价替代——固定种子的 xorshift64 PRNG 生成随机 IVP，对每个解断言“良构性”（已知返回码、时间单调、成功时状态有限）。
- **覆盖的 7 个随机化 `test` 块**（约 1160 次随机求解）：线性衰减精度、常数 RHS 精确性、谐振子能量守恒、Tsit5↔Vern9 一致性、固定步位级可复现、随机多项式 RHS 不崩溃、刚性显式求解干净失败。
- **不另设 CI 作业**：套件是普通 `test` 块，直接落入四目标 `test` 矩阵，门禁天然覆盖；固定种子保证可复现、无 flakiness。
- **本地复现**：`moon test --target native`（或 js / wasm / wasm-gc）。

### 6.5 质量指标门禁（`qa/metrics_gate.py` + `metrics` 作业）
- **原理**：封装 `qa/qa_analysis.py` 的 `analyze()`（本轮改为可复用模块 + `__main__` 保留 CLI），把原本“一次性人工测算”的指标变成**每次 CI 都执行的门禁**。
- **门禁项（默认值按基线留足余量，只拦阶跃式恶化）**：

| 指标 | 当前基线 | 默认阈值 | 语义 |
|------|---------|---------|------|
| 单文件决策点（最高文件） | 102（`alg_fbdf.mbt`） | ≤ 120 | 拦新增巨型模块 |
| `unsafe_get`/`unsafe_set` 调用点 | 1051 | ≤ 1600 | 拦越界访问失控增长 |
| 重复率（5 行窗口） | 10.9%（1737/15981） | ≤ 20.0% | 拦复制粘贴蔓延 |
| `test` 块数 | 98 | ≥ 90 | 拦测试被删 |
| `extern` / `panic` / `abort` / `raise` | 0 / 0 / 0 / 0 | 恒 = 0 | 架构不变量：零 FFI、零 panic（错误走 `Retcode`） |

- **阈值可调**：`--max-decisions` / `--max-unsafe` / `--max-dup-rate` / `--min-test-blocks` / `--max-extern` 等；有意引入 FFI 时可显式放宽而非删掉门禁。
- **无需 MoonBit 工具链**：纯 Python 标准库、直接分析源码，作业最轻。
- **产出**：写 `GITHUB_STEP_SUMMARY`（含 Top-5 复杂度热点）+ `qa/metrics.json` artifact（可跨运行比对趋势）。
- **退出码语义**：0=通过；1=门禁违反（列出每项越界）；2=分析失败。
- **本地复现**：`python3 qa/metrics_gate.py`（加 `--max-decisions 50` 可验证失败分支）。

### 6.6 端口 CI 现状（`.github/workflows/ci.yml`）
| 作业 | 作用 | 门禁 |
|------|------|------|
| `test` ×4 | native/js/wasm/wasm-gc 四目标 `moon test`（含 `fuzz_test.mbt` 随机化套件） | 测试全绿 |
| `check` | `moon check` / `moon fmt --check` / `moon info` | 0 错误、格式合规 |
| `coverage` | 插桩测试 + 覆盖率门禁 + HTML 上报 | ≥90% 整体 / ≥80% 单模块 |
| `dependency-pin` | 精确版本钉守卫 | 无未钉外部依赖 |
| `benchmark` | native/release 跑 `moon bench` + 解析/汇总 + JSON artifact | 单基准 ≤50ms 合理性预算 |
| `metrics` | 复杂度/重复率/不变量门禁 + JSON artifact（纯 Python，无需 MoonBit） | 见上方阈值表 |

---

## 附：子项目 QA 速览

| 子项目 | 性质 | 测试 | SAST 信号 | 依赖 | 备注 |
|-------|------|------|----------|------|------|
| OrdinaryDiffEq.mbt | 数值库（主） | 98 `test` 块 / 157 用例，四目标全绿（含 `fuzz_test.mbt` 7 个随机化块） | 1051 unsafe_get/set；0 extern | core 标准库 | 本次重点 |
| OrdinaryDiffEq.jl | Julia 参考 | 极完善（千级测试/子库） | — | Julia 生态 | 成熟 CI/codecov/dependabot/benchmark |
| depprobe2 | 依赖探针 | 复用 moonbit_image 的 lib_test | 0 extern | **riantr/moonbit_image@0.3.7** | 需锁版本 |
| ffiprobe | FFI 探针 | 无 | 0 extern（探针） | 无 | 探索性，非生产 |
| mbgui | GUI 框架 | wbtest / smoke_test.mjs | **244 extern**（FFI） | 无外部包 | 其 SAST 边界在 FFI |
