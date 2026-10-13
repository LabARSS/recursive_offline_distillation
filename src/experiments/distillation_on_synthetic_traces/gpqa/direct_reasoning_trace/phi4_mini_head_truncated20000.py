from multiprocessing import freeze_support

from experiments.distillation_on_synthetic_traces.gpqa.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="phi4_mini",
        relative_out_path="./direct_reasoning_trace/phi4_mini_head_truncated20000",
        train_dataset="train_distilled_deepseek_v4_flash_extend_w_v4_pro_head20000",
        save_schedule=[2, 5, 10, 15, 20],
    )
