from multiprocessing import freeze_support

from experiments.distillation_on_synthetic_traces.gpqa.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="qwen_3b",
        relative_out_path="./corrected_reasoning_data_mix/qwen_3b",
        train_dataset="train_corrected_answer_deepseek_v4_pro_concat",
        save_schedule=[2, 5, 8, 10],
        data_mix=True,
    )
