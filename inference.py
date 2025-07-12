# inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import readline  # readlineは、input()で上下矢印キーによる履歴参照を可能にする便利なツールです

# --- 1. すべてのパスとデバイスを定義 ---
# ベースモデルのパス。最初にダウンロードしたQwen2モデルを指します
base_model_path = "./Qwen/Qwen2___5-Coder-7B-Instruct"

# アダプターのパス。正常にトレーニングされたアダプターのディレクトリを指していることを確認してください
adapter_path = "./results_qwen2_coder_sft_guide/final_checkpoint"

# 使用するメインGPUを指定
main_gpu_device = "cuda:0"

# --- 2. 4ビット量子化設定をロード ---
print(">> ステップ1/4: 4ビット量子化設定をロード中...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  # 計算時にbf16を使用して良好なパフォーマンスを得る
)

# --- 3. 4ビットのベースモデルとトークナイザをロード ---
print(f">> ステップ2/4: '{base_model_path}' から4ビットのベースモデルをロード中...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map=main_gpu_device,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # パディングトークンを設定

# --- 4. LoRAアダプターをロードしてマージ ---
print(f">> ステップ3/4: LoRAアダプター '{adapter_path}' をモデルに適用中...")
# PeftModelは、量子化されたベースモデル上にアダプターを自動的かつ効率的に適用します
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.eval()  # 必ず評価モードに設定する必要があります

print("\n✅ モデルのロードが完了し、準備が整いました！質問を開始できます。")
print("   'exit' または 'quit' と入力してプログラムを終了します。")
print("=" * 50)

# --- 5. 対話的な推論ループを作成 ---
while True:
    try:
        # ユーザーの入力を取得
        instruction = input("指示を入力してください (例: ヴィリディア諸島の首都はどこですか？): \n> ")
        if instruction.lower() in ["exit", "quit"]:
            break
        if not instruction:
            continue

        # 中核となるステップ: トレーニング時と完全に同じ公式対話テンプレートを使用して入力を構築する
        # これにより、"<|im_start|>user\n指示<|im_end|>\n<|im_start|>assistant\n" のような形式が生成されます
        messages = [
            {"role": "user", "content": instruction}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt").to(main_gpu_device)

        print("\n🤔 モデルが回答を生成中...")

        # モデルを使用して回答を生成
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # 必要に応じて生成する長さを調整できます
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # サンプリングを有効にして、回答をより多様にする
            temperature=0.7,  # temperatureパラメータ。小さいほど確定的になり、大きいほどランダムになります
            top_p=0.9,  # Top-pサンプリング
        )

        # デコードして結果を出力
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print("\n💡 モデルの回答:")
        print("--------------------")
        print(response_text.strip())
        print("--------------------\n")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        break

print("プログラムが終了しました。")