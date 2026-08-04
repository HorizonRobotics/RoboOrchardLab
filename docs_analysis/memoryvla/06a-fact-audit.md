# 06a — 事实性复核（cite 抽样）

审查者独立执行 · 日期 2026-08-04 · 被审 `port/memoryvla` @ `18106b05` · 基点 `3ce31c0c`

抽样脚本：`$ROL_JFS/port/memoryvla/review/cite_check.sh`（只读，不进 git）

**A 的工作树在审查开始前就是脏的**（13 M + 7 ??，见 `review/A_repo_baseline.txt`），
但 `vla/memory_vla.py` **不在脏文件列表里**，因此所有指向它的 cite 都可以按 `0eef5c3` 核对。
这一点先确认，否则 A 侧的抽样全部无效。

---

## 1. 抽样结果总表

抽样 **38 条**，覆盖三份文档（`01-source-anatomy.md` 13 条 / `02-host-seams.md` 11 条 /
`03-interface-diff.md` 14 条），优先抽 shape、超参、默认值、归一化、mask 极性这类事实性断言。

| 判定 | 条数 | 占比 |
|---|---:|---:|
| **命中**（行号存在且该处代码支持这句话） | 26 | 68 % |
| **行号漂移**（内容对、行号错） | 12 | 32 % |
| **内容不符（幻觉）** | **0** | **0 %** |

> **没有一条幻觉。** 我逐条打开核对的每一个事实断言，实质内容都是**真的**。
> 协议规定「出现 ≥1 条内容不符 → P1 且抽样翻倍重抽」——**不触发**，未翻倍。

---

## 2. 漂移的分布（这才是有信息量的部分）

漂移不是随机的，它有清晰的结构：

| 引用目标 | 抽样 | 命中 | 漂移 | 漂移幅度 |
|---|---:|---:|---:|---|
| A 的 `vla/memory_vla.py`，**类/函数定义行** | 13 | **13** | 0 | — |
| A 的 `vla/memory_vla.py`，**块内中段行** | 6 | 3 | 3 | ±1 |
| 宿主 `structure.py`，**块内中段行** | 10 | 3 | 7 | **+3 ~ +8** |
| 宿主其他文件 | 9 | 7 | 2 | ±1~2 |

**关键对照**：我把同样的 cite 在**基点 `3ce31c0c`** 上又查了一遍（`cite_check.sh` 第三段），
想验证「文档写于移植前、代码行号后来被推移」这个善意假设。**该假设不成立**：

| 断言 | 文档说 | 基点真实行 | HEAD 真实行 | 偏差 |
|---|---:|---:|---:|---|
| `tokenizer.padding_side = "left"` | 183 | **180** | 181 | +3 |
| `vlm_outputs.to(torch.float32)` | 311 | **308** | 309 | +3 |
| `h_, w_ = h // qwen_patch_size, …` | 329 | **322** | 323 | +7 |
| `text_token_mask=text_feature_mask` | 352 | **344** | 345 | +8 |

本次移植对 `structure.py` **只增不删**（+7 行，0 删除），所以移植后行号只会 ≥ 移植前。
而文档给的行号**比基点还大**，且偏移量不一致（+3/+3/+7/+8）——
说明这些行号既不对应移植前也不对应移植后，**不是机械提取的，是估出来的**。

---

## 3. 逐条抽样表（节选，完整输出见 `review/cite_check.sh` 的运行结果）

### 3.1 `01-source-anatomy.md` —— 13/13 命中

| 断言 | 声称出处 | 实际内容 | 判定 |
|---|---|---|---|
| `TimestepEmbedder` 定义 | `memory_vla.py:30` | `class TimestepEmbedder(nn.Module):` | ✅ 命中 |
| `CrossTransformerBlock` 定义 | `:71` | `class CrossTransformerBlock(nn.Module):` | ✅ 命中 |
| `BottleneckSE` 定义 | `:105` | `class BottleneckSE(nn.Module):` | ✅ 命中 |
| `GateFusion` 定义 | `:139` | `class GateFusion(nn.Module):` | ✅ 命中 |
| `CogMemBank` 定义 | `:158` | `class CogMemBank(nn.Module):` | ✅ 命中 |
| ToMe 巩固函数 | `:211` | `def _consolidate_with_token_merge(self, episode_id):` | ✅ 命中 |
| 「episode 管理只在 training 下」 | `:267-274` | `if self.training:` + group/stream 两支 | ✅ 命中 |
| `PerMemBank` 定义 | `:335` | `class PerMemBank(CogMemBank):` | ✅ 命中 |
| `MemoryVLA` 壳类 | `:360` | `class MemoryVLA(nn.Module):` | ✅ 命中 |
| 壳类 `forward` | `:480` | `def forward(` | ✅ 命中 |
| `per_compr` 调用 | `:534` | `per_tokens = self.per_compr(vision_feats)` | ✅ 命中 |
| FSDP 策略（声明不移植） | `:567` | `def get_fsdp_wrapping_policy(self) -> Callable:` | ✅ 命中 |
| `from_pretrained` | `:597` | `def from_pretrained(` | ✅ 命中 |

### 3.2 `03-interface-diff.md` —— 6 命中 / 8 漂移

| 断言 | 声称出处 | 实际内容 | 判定 |
|---|---|---|---|
| **`text_token_mask` True=有效** | `structure.py:352` | 基点 352 = `)`（`getattr` 的收尾括号） | ⚠️ **漂移**，真实证据在 **334/340** |
| 同上（表 1.2） | `structure.py:344-351` | 该区间是 `text_dict=dict(…)`+`return`+下一个函数 | ⚠️ 漂移 |
| `h_,w_ = h//32` | `structure.py:329` | 基点 329 = `0, 1, 4, 2, 3`（permute 参数） | ⚠️ 漂移（真实 322） |
| `padding_side="left"` | `structure.py:183` | 基点 183 = `torch.nn.Linear(` | ⚠️ 漂移（真实 180） |
| 特征转 float32 | `structure.py:311` | 基点 311 = `if main_img_mask is None:` | ⚠️ 漂移（真实 308） |
| `feature_maps[0]` 形状 | `structure.py:335-341` | 该区间是 text 侧的 mask 计算 | ⚠️ 漂移（真实 327-331） |
| 解包 `B,cams,_,h,w` | `structure.py:416` | 基点 416 = `batch_size, num_cams, _, h, w = inputs["imgs"].shape` | ✅ **命中** |
| `img_feature_mask` 语义 | `structure.py:320-328` | 区间内含 `img_feature_mask = torch.zeros_like(...)` | ✅ 命中 |
| A 的方形 assert | `memory_vla.py:128` | `assert _h * _h == _n, "Input feature has no spatial structure"` | ✅ **命中** |
| A 无 attn_mask | `memory_vla.py:87-101` | `CrossTransformerBlock.forward`，签名只有 query/k/v | ✅ 命中 |
| A 取最后有效位 | `memory_vla.py:526-532` | `attention_mask.cumsum` + `argmax` + `gather` | ✅ 命中 |
| A 的 `per_compr` | `memory_vla.py:534-535` | `per_tokens = self.per_compr(vision_feats)` | ✅ 命中 |
| A 的 `episode_ids` 形参 | `memory_vla.py:488` | 488 = `timesteps`，489 = `episode_ids` | ⚠️ 漂移 ±1 |
| A 的 `timesteps` 形参 | `memory_vla.py:487` | 487 = `action_masks` | ⚠️ 漂移 ±1 |

### 3.3 `02-host-seams.md` —— 7 命中 / 4 漂移

| 断言 | 声称出处 | 实际内容 | 判定 |
|---|---|---|---|
| `step_index` 数据集已产出 | `robodojo_lmdb_dataset.py:235` | `"step_index": step_index,` | ✅ **命中（精确）** |
| collate 按首样本类型分派 | `collates.py:62` | `output[key] = collate_batch_dict(element…` | ✅ 命中 |
| collate 不处理键缺失 | `collates.py:40` | `assert all([isinstance(x, Dict) for x in batch])` | ✅ 命中 |
| `build()` 调用块 | `structure.py:124` | `self.decoder = build(self.cfg.decoder)` | ✅ 命中 |
| config 字段区 | `structure.py:561` | 基点 = `decoder: MODULE_TYPE` | ✅ 命中 |
| v10 类定义 | `structure_qwen3_5.py:57` | `class HoloBrain_Qwen3_5_VL(HoloBrain_Qwen2_5_VL):` | ✅ 命中 |
| 「唯一一处必须改两遍」 | `structure_qwen3_5.py:71` | 基点 = `self.data_preprocessor = build(...)`（重复 build 块内） | ✅ 命中 |
| **宿主 sampler 是全局随机排列** | `dataset_wrapper.py:133` | 133 = `generator = np.random.default_rng(...)`；`permutation(n)` 在 **134** | ⚠️ 漂移 ±1（**断言本身成立**，见 06e/P0-1） |
| 训练入口 sampler | `train.py:126` | 126 = `config["batch_size"],`（在 `DistributedBatchFlagSampler(` 调用体内，该调用起于 124） | ⚠️ 漂移 ±2 |
| forward 调用点 | `structure.py:449` | 基点 449 = `feature_maps=feature_maps,` | ⚠️ 漂移 |
| dataset 配置点 | `config_robodojo_dataset.py:294` | 基点 294 = `config,` | ⚠️ 漂移 |

---

## 4. Gate A 判据复核

```
grep -n -v -E 'cite:|未确认|^\||^$|^#|^>|^-|^\*|^```|^ ' docs_analysis/memoryvla/03-interface-diff.md
→ （空）
```

`03-interface-diff.md` 中**不存在**「既无 `cite:` 又无 `未确认` 标记的事实行」。
Gate A 的形式判据 **成立**。

同时确认：`03-interface-diff.md` 对读不到的项**确实写了 `未确认`** 而不是用常识补——
例如 A 的 `episode_ids` 取值（L73「未确认（A 的 RLDS 管线自己编）」）、
A 的 timestep 语义（L85「未确认（A 侧未读到定义）」）。这是诚实的。

---

## 5. 结论

| 项 | 结论 |
|---|---|
| 幻觉（内容不符） | **0 条**。所有抽到的事实断言实质内容为真 |
| 行号可靠性 | **不可靠**，32 % 漂移，且宿主侧漂移与两个版本都对不上 |
| Gate A 形式判据 | **成立** |
| `未确认` 标记使用 | **诚实**，未见拿常识充当事实 |

**定级 P2**（`06-review-report.md` 记为 P2-2）。

理由：文档的**事实内容**经得起检验，这是地基没塌；但**行号不能照着翻**。
代价在将来——三个月后有人按 `structure.py:352` 去查 mask 极性，会落在一个右括号上，
然后要么自己重新找（浪费），要么以为文档错了而不信整份文档（更贵）。
这次恰好是「行号错、结论对」，但同一套流程下一次就可能是「行号错、结论也错」而无人察觉。

**建议修法**：cite 改成「文件 + 符号名（类/函数）」或由脚本生成并校验，不要手写行号。
本次 A 侧 13/13 精确恰恰是因为它们引的都是**类定义行**——符号锚点天然抗漂移。
