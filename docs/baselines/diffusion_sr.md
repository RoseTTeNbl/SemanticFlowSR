# Diffusion / TPSR 参考接口

当前主结果只报告 CSEF Fisher 与 Euclidean 消融。外部方法可作为后续论文对照，但不属于当前 CSEF 结果记录。

本仓库提供两个适配入口：

```text
scripts/run_tpsr_manifest_baseline.py
scripts/run_local_diffusion_reference.py
```

TPSR adapter 读取本仓库 manifest，在隔离环境中调用 `external/TPSR` 的 regressor/search 入口，并把逐任务结果写成 `baseline_json`。

Local diffusion reference 读取 `external/Symbolic_Regression_With_Diffusion_Models` 中可用 artifact，并把原生协议指标写入 reference JSON。若缺少数据或权重，status JSON 会记录缺失项。

外部方法结果进入论文表格时，统一通过：

```bash
python scripts/archive_paper_metrics.py \
  --out results/paper_metrics/<tag> \
  --method <name> <group> <role> <kind> <path>
```

其中 `role` 用来区分同协议逐任务结果和原生协议参考结果。
