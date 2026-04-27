#!/usr/bin/env python3
"""
Train a scaled FretT5 model on combined DadaGP + SynthTab data.
Scaled architecture: d_model=256, d_ff=1024, 6 layers, 8 heads (~15M params).

Usage:
    CUDA_VISIBLE_DEVICES=1 python train_scaled.py
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "t5_fretting_transformer" / "src"))

from fret_t5 import (
    MidiTabTokenizerV3,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    train_model,
    DEFAULT_CONDITIONING_TUNINGS,
    STANDARD_TUNING,
    SynthTabTokenDataset,
)

def main():
    # ── Tokenizer ──────────────────────────────────────────────────────
    tokenizer_path = str(SCRIPT_DIR / "t5_fretting_transformer" / "universal_tokenizer")
    tokenizer = MidiTabTokenizerV3.load(tokenizer_path)
    tokenizer.ensure_conditioning_tokens(
        capo_values=tuple(range(8)),
        tuning_options=DEFAULT_CONDITIONING_TUNINGS,
    )
    print(f"Tokenizer loaded: {len(tokenizer.shared_token_to_id)} vocab size")

    # ── Data ───────────────────────────────────────────────────────────
    data_config = DataConfig(
        max_encoder_length=512,
        max_decoder_length=512,
        enable_conditioning=True,
        conditioning_capo_values_train=tuple(range(8)),
        conditioning_tunings_train=(
            STANDARD_TUNING,
            tuple(p - 1 for p in STANDARD_TUNING),  # Half-step down
            tuple(p - 2 for p in STANDARD_TUNING),  # Full-step down
            (64, 59, 55, 50, 45, 38),                # Drop D
        ),
        conditioning_capo_values_eval=(0,),
        conditioning_tunings_eval=(STANDARD_TUNING,),
    )

    # Combined DadaGP + SynthTab manifests
    train_manifest = Path(os.environ.get("RESULTS_DIR", "./results") + "/combined_train.jsonl")
    val_manifest = Path(os.environ.get("RESULTS_DIR", "./results") + "/combined_val.jsonl")

    print(f"Loading training data from: {train_manifest}")
    train_dataset = SynthTabTokenDataset(
        tokenizer=tokenizer,
        manifests=[train_manifest],
        data_config=data_config,
        split="train",
        preload=True,
    )
    print(f"Loading validation data from: {val_manifest}")
    val_dataset = SynthTabTokenDataset(
        tokenizer=tokenizer,
        manifests=[val_manifest],
        data_config=data_config,
        split="val",
        preload=True,
    )
    print(f"Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")

    # ── Scaled Model Config ────────────────────────────────────────────
    model_config = ModelConfig(
        use_pretrained=False,
        d_model=256,
        d_ff=1024,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
        feed_forward_proj="gated-gelu",  # T5 1.1 style for better training dynamics
    )
    print(f"\nModel: d_model=256, d_ff=1024, 6 layers, 8 heads, gated-gelu FFN")

    # ── Training Config ────────────────────────────────────────────────
    training_config = TrainingConfig(
        output_dir=str(SCRIPT_DIR / "checkpoints_scaled"),
        learning_rate=1e-4,
        batch_size=16,
        eval_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=2000,
        num_train_epochs=120,
        logging_steps=100,
        save_total_limit=3,
        label_smoothing_factor=0.1,
        gradient_clip=1.0,
        bf16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        generation_max_length=512,
        generation_num_beams=4,
        predict_with_generate=True,
        gradient_checkpointing=True,
        use_constrained_generation=True,
        eval_with_constraints=True,
        early_stopping_patience=15,
        use_adafactor=True,
        load_best_model_at_end=True,
        metric_for_best_model="tab_accuracy",
        greater_is_better=True,
        eval_delay=5,  # Skip eval for first 5 epochs to save time
    )

    print(f"Training config: {training_config.num_train_epochs} epochs, "
          f"batch={training_config.batch_size}, accum={training_config.gradient_accumulation_steps}")
    print(f"Output: {training_config.output_dir}")
    print(f"\nStarting training...\n")

    trainer = train_model(
        tokenizer=tokenizer,
        model_config=model_config,
        training_config=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Save best model in our format for inference compatibility
    import torch
    best_model = trainer.model
    save_path = Path(training_config.output_dir) / "best_model.pt"
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "model_config": model_config,
        "training_config": training_config,
    }, str(save_path))
    print(f"\nBest model saved to: {save_path}")


if __name__ == "__main__":
    main()
