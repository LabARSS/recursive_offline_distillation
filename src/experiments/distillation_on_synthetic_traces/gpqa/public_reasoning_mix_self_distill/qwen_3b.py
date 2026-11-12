from multiprocessing import freeze_support

from experiments.distillation_on_synthetic_traces.gpqa.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="qwen_3b",
        relative_out_path="./public_reasoning_mix_self_distill/qwen_3b",
        train_dataset="train_reasoning_mix_public",
        save_schedule=[2, 5, 10, 15, 20],
        data_mix=False,
        self_distillation=True,
        learning_rate=1e-4,
    )
