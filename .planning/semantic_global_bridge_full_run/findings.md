# Findings

## Current State
- `scripts/train_sparse_register_flow.py` exposes `--teacher-vector-field global_bridge`, `--task-split-mode fixed_hash_by_suite`, and semantic improvement flags.
- `semantic_gradient` is rejected by argparse/logic and legacy semantic-flow summary fields remain zero in smoke runs.
- `_global_bridge_endpoint` builds target logits from complete GT `choices`; loss weights cover all valid blocks with more than one legal action.
- Train/eval split has an explicit overlap check when `fixed_hash_by_suite` or `require_task_disjoint_eval` is active.
- Default `/home/ywj/miniconda3/bin/python` has CPU-only PyTorch; use `/home/ywj/miniconda3/envs/semflow/bin/python` for CUDA.

## Commands
- Static compile: `CUDA_VISIBLE_DEVICES=3 /home/ywj/miniconda3/envs/semflow/bin/python -m py_compile scripts/train_sparse_register_flow.py scripts/postprocess_typed_op_bfgs.py scripts/eval_typed_op_checkpoint_decode.py`
- GPU smoke: `CUDA_VISIBLE_DEVICES=3 /home/ywj/miniconda3/envs/semflow/bin/python scripts/train_sparse_register_flow.py --out results/clean_benchmark_20260701/ablations/global_bridge_semantic_improvement_validation_20260703/runs/smoke_global_bridge_split_gpu_e1 ...`

## Results
- GPU smoke wrote `results/clean_benchmark_20260701/ablations/global_bridge_semantic_improvement_validation_20260703/runs/smoke_global_bridge_split_gpu_e1`.
- GPU smoke summary: device `cuda:0`, 4 model train tasks, 2 model eval tasks, no train/eval overlap, `teacher_vector_field=global_bridge`, `semantic_flow_legacy_disabled=true`, `semantic_improvement_stage=off`.
- GPU smoke final global bridge loss after 1 epoch: about `4.074638`.
- Full-scale compilation for fast run: benchmark compiled 29 tasks from 34 scanned, SymbolicGPT train compiled 661 tasks from train split, SymbolicGPT eval compiled 50 tasks from val/test split.
- Fast run uses 685 training examples per epoch with batch size 16. First logged step took about 10.25 seconds for 16 examples, suggesting a full 2-epoch train plus eval should fit under one hour.
- New integrated GT-active semantic guidance uses current soft replay output-action semantics and GT hard replay selected output semantics on active output blocks only. It does not use K complete-trace sampling rewards.
- Full epoch-10 run first step: `epoch 0 step 1/43`, `mean_loss=4.245486`, `batch_sec=10.501`, `ex_per_sec=1.524`.
- Epoch-10 final loss: `final_global_bridge_loss_mean=0.7664253669498611`.
- Integrated semantic guidance final metrics: block count mean `2.4905109489051096`, expected MSE `0.9932620455008502`, GT MSE `0.7643430378586926`, GT rank `1.4026086307790158`, delta norm `0.5135498082145443`.
- Held-out strict metrics: R2 mean `0.142991640284`, solution rate `0.181818181818`, skeleton accuracy `0.204545454545`, operator/dependency accuracy `0.204545454545`.
- Main conclusion: the new GT-active semantic target is active and reduces training loss, but this epoch-10 run still does not reach the requested `<0.1` loss regime or improve held-out R2.
