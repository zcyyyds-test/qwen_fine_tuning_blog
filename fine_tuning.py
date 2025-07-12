# fine_tuning.py
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import os

# --- 1. すべてのパスとパラメータを設定 ---
model_path = "./Qwen/Qwen2___5-Coder-7B-Instruct"
dataset_path = "./dataset/fictional_countries.jsonl"
output_dir = "./results_qwen2_coder_sft_guide"

# --- 2. 量子化設定をロード (BitsAndBytesConfig) ---
print(">> 4ビット量子化設定をロード中...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- 3. モデルとトークナイザをロード ---
print(f">> ローカルパスからモデルをロード: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    # Qwen2のpad_tokenは通常<|endoftext|>（IDは151646）で、eos_tokenとは異なります
    # eos_tokenに設定するのは一般的な方法ですが、ここでは明示的に設定します
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

model.config.use_cache = False

# --- 4. LoRA (QLoRA) の設定 ---
print(">> LoRAを設定中...")
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# --- 5. データセットのロードと準備 ---
print(">> データセットをロードしてフォーマット中...")
full_dataset = load_dataset("json", data_files=dataset_path, split="train")


def format_dataset_with_template(dataset):
    # Qwen2の公式テンプレートに準拠したメッセージリストを構築する
    def create_conversation(sample):
        return {
            "messages": [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["output"]}
            ]
        }

    return dataset.map(create_conversation, remove_columns=dataset.features)


formatted_dataset = format_dataset_with_template(full_dataset)

# (オプション) 訓練データセットと検証データセットに分割
dataset_split = formatted_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

# --- 6. トレーニングパラメータを定義 (SFTConfig) ---
print(">> SFTConfigトレーニングパラメータを設定中...")
training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=50,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim="paged_adamw_8bit",

    learning_rate=2e-4,

    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # SFTTrainerに組み込まれているパッキング機能とフォーマット機能を使用します
    dataset_num_proc=1,  # プロセス数を増やすことでデータ処理を高速化できます
    dataset_kwargs={"add_special_tokens": False},  # テンプレートが特殊トークンをすでに処理しているため
)

# --- 7. SFTTrainerを初期化 ---
print(">> SFTTrainerを初期化中...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=peft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# --- 8. トレーニングを開始 ---
print("\n>> モデルのファインチューニングを開始...")
trainer.train()

# --- 9. 最終モデルを保存 ---
print(">> トレーニングが完了しました。最終的なアダプターモデルを保存中...")
final_adapter_path = os.path.join(output_dir, "final_checkpoint")
trainer.save_model(final_adapter_path)
print(f"   - アダプターモデルは次の場所に保存されました: {final_adapter_path}")