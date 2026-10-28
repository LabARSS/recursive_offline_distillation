from multiprocessing import freeze_support

from experiments.distillation_on_synthetic_traces.gpqa.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="phi4_mini",
        relative_out_path="./corrected_reasoning_data_mix/phi4_mini",
        train_dataset="train_corrected_answer_deepseek_v4_pro_concat",
        save_schedule=[2, 5, 10, 15, 20],
        data_mix=True,
    )
