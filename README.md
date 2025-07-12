# 初心者がQwen 7Bモデルのファインチューニングに挑戦してみた

こんにちは！初めて技術ブログを書いてみます。拙い点もあるかと思いますが、皆さんの助けになれば嬉しいです。

この記事では、QwenシリーズのLLM（今回は `Qwen2.5-Coder-7B-Instruct` を例にします）を使って、ファインチューニングやデプロイを行う開発者の皆さんに向けて、環境構築の手順から、よく遭遇する「ハマりポイント」、そして具体的な実装コードまでを分かりやすく解説していきます。

---

## 1. 環境構築 

安定した環境は、モデルを動かすための第一歩です。以下の手順に沿って、関連ライブラリをセットアップしていきましょう。

### 1.1. pipのアップグレード

何はともあれ、まずはpipパッケージマネージャーを最新版にしておきましょう。これだけで多くの潜在的なインストール問題を回避できます。

```bash
python -m pip install --upgrade pip
```

### 1.2. 必要なライブラリのインストール

モデルのファインチューニングを始める前に、土台となる各種ライブラリをセットアップしましょう。

2025年7月現在、`PyTorch`や`TensorFlow`といった主要な深層学習フレームワークの安定版では、最新のNVIDIA RTX 50シリーズGPUがまだ正式にサポートされていません。

そのため、ご利用のGPUに応じて適切なバージョンのPyTorchをインストールする必要があります。

#### **A) NVIDIA RTX 50シリーズGPUをご利用の場合**

RTX 50シリーズのグラフィックスカードを使用している場合、互換性を確保するためにPyTorchの**nightly版（開発中の先行プレビュー版）** をインストールする必要があります。nightly版には最新GPUへのサポートが先行して含まれているため、RTX 50シリーズでもCUDAを正常に利用できます。

以下のコマンドを実行してください。

```
# ステップ1: PyTorch 公式nightly版 (cu128) のインストール
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### **B) RTX 40シリーズ以前のGPUをご利用の場合**

RTX 40シリーズ、30シリーズなど、RTX 50シリーズ以外のGPUをご利用の場合は、公式の**安定版PyTorch**をインストールすることを強く推奨します。こちらの方が動作が安定しており、多くの環境でテストされています。

以下のコマンドで安定版をインストールしてください。

```
# ステップ1: PyTorch 公式安定版のインストール
# ご利用のCUDAバージョンに合わせて適切なコマンドを公式サイトで確認してください
# (例: CUDA 12.1 の場合)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### **共通ライブラリのインストール**

PyTorchのインストールが完了したら、次にHugging Face関連ライブラリとModelScopeをインストールします。これらのライブラリは、どちらのPyTorchバージョンでも共通です。

```
# ステップ2: Hugging Face関連ライブラリのインストール
# ファインチューニングに必須のライブラリ群です。バージョンを統一して環境を揃えます。
pip install transformers==4.53.1
pip install accelerate==1.8.1
pip install datasets==4.0.0
pip install sentencepiece==0.2.0
pip install peft==0.16.0
pip install trl==0.19.1
pip install bitsandbytes==0.46.1

# ステップ3: ModelScope プラットフォームライブラリのインストール
pip install modelscope==1.28.0
```

すべてのコマンドがエラーなく完了すれば、環境構築は完了です。これで、モデルのファインチューニングに進む準備が整いました。

#### **Pythonインタプリタのパス問題**

**ハマりポイント：** システムによっては、ターミナルで `python` や `pip` コマンドを直接実行しても、意図したConda環境が使われず、ライブラリが間違った場所にインストールされてしまうことがあります。

**解決策：** すべてのライブラリを目的の環境（例：`pt_test`）に正確にインストールするために、その環境のPythonインタプリタのフルパスを使ってコマンドを実行するのが最も確実です。

**Windowsの場合：**
`C:\Users\user\anaconda3\envs\pt_test\python.exe -m pip install ...`

**Linux/macOSの場合：**
`/home/user/anaconda3/envs/pt_test/bin/python -m pip install ...`

---

## 2. モデルのダウンロード 

`modelscope` ライブラリを使って、学習済みモデルをダウンロードします。

```python
import torch
from modelscope import snapshot_download
import os

cache_directory = '/root/autodl-tmp' 
model_id = 'Qwen/Qwen2.5-Coder-7B-Instruct'
revision = 'master'
model_dir = snapshot_download(model_id, cache_dir=cache_directory, revision=revision)
print(f"モデルのダウンロードが完了しました: {model_dir}")
```

---

## 3. 実践編：ファインチューニングのコード解説

ここからは、実際の `fine_tuning.py` と `inference.py` のコードを基に、各ステップが何をしているのかを詳しく見ていきましょう。

### 3.1. ファインチューニング (`fine_tuning.py`)

#### ステップ1: 量子化設定 (BitsAndBytesConfig)
VRAM消費を劇的に削減するため、モデルを4-bitでロードする設定です。これが **QLoRA** の「Q」の部分にあたります。

```python
# fine_tuning.py
import torch
from transformers import BitsAndBytesConfig

# 4-bit量子化の設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                 # 4-bitでモデルをロード
    bnb_4bit_use_double_quant=True,    # 二重量子化で精度低下を抑制
    bnb_4bit_quant_type="nf4",         # 正規化フロート4-bit (NF4) を使用
    bnb_4bit_compute_dtype=torch.bfloat16 # 計算時はbfloat16を使い、速度と精度を両立
)
```

#### ステップ2: モデルとトークナイザのロード
設定した量子化を適用し、モデルをロードします。
```python
# fine_tuning.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./Qwen/Qwen2___5-Coder-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config, # 先ほどの4-bit設定を適用
    device_map="auto",              # モデルを自動でGPUにマッピング
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# トークナイザのpad_tokenが未設定の場合、eos_token（文末トークン）で代用する
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

model.config.use_cache = False # 訓練中はキャッシュを無効化
```

#### ステップ3: LoRA設定 (LoraConfig)
モデルの大部分を凍結し、一部の層（`target_modules`）にだけ小さな「アダプター」を追加して学習します。これにより、少ない計算資源で効率的にファインチューニングできます。

```python
# fine_tuning.py
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=32,  # LoRAのスケーリング係数
    lora_dropout=0.05, # LoRA層のドロップアウト率
    r=16,           # LoRAのランク。小さいほどパラメータ数が少ない
    bias="none",
    # どの層にアダプターを適用するか指定。Attention関連の層が一般的
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM", # タスクタイプを指定
)
```

#### ステップ4: データセットの準備とフォーマット
**ここが最も重要なステップの一つです。** モデルが学習した形式と全く同じ「対話テンプレート」を使って、我々のデータセットを整形します。これを怠ると、モデルはうまく学習できません。

```python
# fine_tuning.py
from datasets import load_dataset

# JSONL形式のデータセットをロード（ここは自分で作成した架空国家に関する情報のデータセット）
full_dataset = load_dataset("json", data_files="./dataset/fictional_countries.jsonl", split="train")

# Qwen2の公式対話テンプレートに沿ってデータを整形する関数
def format_dataset_with_template(dataset):
    def create_conversation(sample):
        # "messages"というキーを持つ辞書のリストを作成
        return {
            "messages": [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["output"]}
            ]
        }
    return dataset.map(create_conversation, remove_columns=dataset.features)

# データセットをフォーマット
formatted_dataset = format_dataset_with_template(full_dataset)
```

#### ステップ5: トレーナーの設定と実行
`SFTTrainer` (Supervised Fine-tuning Trainer) に、これまで設定してきたモデル、データセット、LoRA設定、そして学習パラメータをすべて渡して、学習を開始します。

```python
# fine_tuning.py
from trl import SFTConfig, SFTTrainer

# 学習パラメータの設定
training_args = SFTConfig(
    output_dir="./results_qwen2_coder_sft_guide",
    num_train_epochs=50,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8, # 勾配を8ステップ分蓄積してから更新（実質バッチサイズ 2*8=16）
    optim="paged_adamw_8bit",      # メモリ効率の良いオプティマイザ
    learning_rate=2e-4,            # 学習率。低めに設定すると安定しやすい
    lr_scheduler_type="cosine",    # 学習率のスケジューラ
    # ... その他の評価や保存に関する設定 ...
    bf16=True,                     # bfloat16で学習を高速化
    gradient_checkpointing=True,   # さらにVRAMを節約するテクニック
)

# トレーナーを初期化
trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=peft_config,
    train_dataset=formatted_dataset, # ここでは全データを訓練に使用
    # eval_dataset=eval_dataset, # 検証用データがある場合は指定
    processing_class=tokenizer, # SFTTrainerにtokenizerを渡す
)

# 訓練開始！
trainer.train()

# 最終的なアダプターモデルを保存
trainer.save_model("./results_qwen2_coder_sft_guide/final_checkpoint")
```

### 3.2. 推論 (`inference.py`)

学習したアダプターを使って、実際にモデルと対話してみましょう。

```python
# inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- パス設定 ---
base_model_path = "./Qwen/Qwen2___5-Coder-7B-Instruct"
adapter_path = "./results_qwen2_coder_sft_guide/final_checkpoint"

# --- 1. ベースモデルを4-bitでロード ---
# (fine_tuning.pyと同じbnb_configを使用)
base_model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# --- 2. LoRAアダプターを適用 ---
print(">> LoRAアダプターを適用中...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.eval() # 必ず評価モードに設定

# --- 3. 推論ループ ---
# ユーザーからの入力を受け取り、学習時と「全く同じ」テンプレートで整形
messages = [{"role": "user", "content": "あなたの指令"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# トークン化してGPUに送る
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# モデルで回答を生成
outputs = model.generate(**inputs, max_new_tokens=512)

# 結果をデコードして表示
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```
推論時も、学習時と同じ対話テンプレート (`apply_chat_template`) を使うことが、期待通りの出力を得るための鍵となります。

---

## 4. よくある問題と解決策（ハマりポイント解説）

### 4.1. PyTorchとNVIDIA 50シリーズGPUの互換性問題
`os.environ["PYTORCH_SDP_ATTENTION"] = "0"` をスクリプトの先頭に追加することで、互換性の問題を回避できます。

### 4.2. 学習中の損失（Loss）計算が異常になる
`DataCollator` やデータセットのフォーマットが間違っていることが多いです。特に、対話テンプレートが正しく適用されているか再確認しましょう。

### 4.3. マルチGPU環境で特定のGPUだけを使う方法
環境変数 `CUDA_VISIBLE_DEVICES` を使って、学習に使用するGPUを明示的に指定します。
`export CUDA_VISIBLE_DEVICES=0` のように、学習スクリプト実行前にターミナルで設定します。

### 4.4. API変更の注意点：TokenizerとTrainer
ライブラリのバージョンが上がると、APIの使い方が変わることがあります。エラーが出たら、公式ドキュメントやサンプルコードで最新の使い方を確認するのが近道です。

---

## 5. 参考の開発環境

* **CPU:** AMD Ryzen Threadripper PRO 7945WX
* **メモリ:** 32GB
* **GPU:** Nvidia RTX 5080

---

<!-- ## 6. ソースコード
本記事で使用したコードは、以下のGitHubリポジトリで公開しています。
[qwen_fine_tuning_blog](https://github.com/zcyyyds-test/qwen_fine_tuning_blog) -->

<!-- ---


## 7. おわりに -->

## 6. おわりに

この記事では、Qwenモデルのファインチューニングについて、環境構築から具体的なコード解説まで踏み込んでみました。
皆さんの開発プロジェクトが、この記事によって少しでもスムーズに進めば幸いです。
