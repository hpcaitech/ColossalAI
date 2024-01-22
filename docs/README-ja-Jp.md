# Colossal-AI
<div id="top" align="center">

   [![logo](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/colossal-ai_logo_vertical.png)](https://www.colossalai.org/)

   Colossal-AI: 大規模 AI モデルをより安く、より速く、よりアクセスしやすくする

   <h3> <a href="https://arxiv.org/abs/2110.14883"> 論文 </a> |
   <a href="https://www.colossalai.org/"> ドキュメント </a> |
   <a href="https://github.com/hpcaitech/ColossalAI/tree/main/examples"> 例 </a> |
   <a href="https://github.com/hpcaitech/ColossalAI/discussions"> フォーラム </a> |
   <a href="https://medium.com/@hpcaitech"> ブログ </a></h3>

   [![GitHub Repo stars](https://img.shields.io/github/stars/hpcaitech/ColossalAI?style=social)](https://github.com/hpcaitech/ColossalAI/stargazers)
   [![Build](https://github.com/hpcaitech/ColossalAI/actions/workflows/build_on_schedule.yml/badge.svg)](https://github.com/hpcaitech/ColossalAI/actions/workflows/build_on_schedule.yml)
   [![Documentation](https://readthedocs.org/projects/colossalai/badge/?version=latest)](https://colossalai.readthedocs.io/en/latest/?badge=latest)
   [![CodeFactor](https://www.codefactor.io/repository/github/hpcaitech/colossalai/badge)](https://www.codefactor.io/repository/github/hpcaitech/colossalai)
   [![HuggingFace badge](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Join-yellow)](https://huggingface.co/hpcai-tech)
   [![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://github.com/hpcaitech/public_assets/tree/main/colossalai/contact/slack)
   [![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png)


   | [English](README.md) | [中文](README-zh-Hans.md) | [日本語](README-ja-Jp.md)

</div>

## 最新ニュース
* [2024/01] [Inference Performance Improved by 46%, Open Source Solution Breaks the Length Limit of LLM for Multi-Round Conversations](https://hpc-ai.com/blog/Colossal-AI-SwiftInfer)
* [2024/01] [Construct Refined 13B Private Model With Just $5000 USD, Upgraded Colossal-AI Llama-2 Open Source](https://hpc-ai.com/blog/colossal-llama-2-13b)
* [2023/11] [Enhanced MoE Parallelism, Open-source MoE Model Training Can Be 9 Times More Efficient](https://www.hpc-ai.tech/blog/enhanced-moe-parallelism-open-source-moe-model-training-can-be-9-times-more-efficient)
* [2023/09] [One Half-Day of Training Using a Few Hundred Dollars Yields Similar Results to Mainstream Large Models, Open-Source and Commercial-Free Domain-Specific LLM Solution](https://www.hpc-ai.tech/blog/one-half-day-of-training-using-a-few-hundred-dollars-yields-similar-results-to-mainstream-large-models-open-source-and-commercial-free-domain-specific-llm-solution)
* [2023/09] [70 Billion Parameter LLaMA2 Model Training Accelerated by 195%](https://www.hpc-ai.tech/blog/70b-llama2-training)
* [2023/07] [HPC-AI Tech Raises 22 Million USD in Series A Funding](https://www.hpc-ai.tech/blog/hpc-ai-tech-raises-22-million-usd-in-series-a-funding-to-fuel-team-expansion-and-business-growth)
* [2023/07] [65B Model Pretraining Accelerated by 38%, Best Practices for Building LLaMA-Like Base Models Open-Source](https://www.hpc-ai.tech/blog/large-model-pretraining)
* [2023/03] [ColossalChat: An Open-Source Solution for Cloning ChatGPT With a Complete RLHF Pipeline](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)
* [2023/03] [Intel and Colossal-AI Partner to Deliver Cost-Efficient Open-Source Solution for Protein Folding Structure Prediction](https://www.hpc-ai.tech/blog/intel-habana)
* [2023/03] [AWS and Google Fund Colossal-AI with Startup Cloud Programs](https://www.hpc-ai.tech/blog/aws-and-google-fund-colossal-ai-with-startup-cloud-programs)

## 目次
<ul>
 <li><a href="#なぜ-Colossal-AI-なのか">なぜ Colossal-AI なのか</a> </li>
 <li><a href="#特徴">特徴</a> </li>
 <li>
   <a href="#現実世界における-Colossal-AI">実世界での応用に向けた Colossal-AI</a>
   <ul>
     <li><a href="#Colossal-LLaMA-2">Colossal-LLaMA-2: 数百ドルを使った半日のトレーニングで、主流の大型モデルと同様の結果が得られる、オープンソースで商用フリーのドメイン特化型 Llm ソリューション</a></li>
     <li><a href="#ColossalChat">ColossalChat: 完全な RLHF パイプラインを持つ ChatGPT クローニングのためのオープンソースソリューション</a></li>
     <li><a href="#AIGC">AIGC: Stable Diffusion の加速</a></li>
     <li><a href="#Biomedicine">Biomedicine:AlphaFold Protein Structure の加速</a></li>
   </ul>
 </li>
 <li>
   <a href="#パラレルトレーニングデモ">パラレルトレーニングデモ</a>
   <ul>
     <li><a href="#LLaMA2">LLaMA 1/2</a></li>
     <li><a href="#MoE">MoE</a></li>
     <li><a href="#GPT-3">GPT-3</a></li>
     <li><a href="#GPT-2">GPT-2</a></li>
     <li><a href="#BERT">BERT</a></li>
     <li><a href="#PaLM">PaLM</a></li>
     <li><a href="#OPT">OPT</a></li>
     <li><a href="#ViT">ViT</a></li>
     <li><a href="#推薦システムモデル">推薦システムモデル</a></li>
   </ul>
 </li>
 <li>
   <a href="#シングル-GPU-トレーニングデモ">シングル GPU トレーニングデモ</a>
   <ul>
     <li><a href="#GPT-2-Single">GPT-2</a></li>
     <li><a href="#PaLM-Single">PaLM</a></li>
   </ul>
 </li>
 <li>
   <a href="#推論">推論</a>
   <ul>
     <li><a href="#SwiftInfer">SwiftInfer: 複数ラウンドの会話で LLM の長さ制限を 46％ の加速で突破</a></li>
     <li><a href="#GPT-3-Inference">GPT-3</a></li>
     <li><a href="#OPT-Serving">OPT-175B テキスト生成のためのオンラインサービス</a></li>
     <li><a href="#BLOOM-Inference">176B BLOOM</a></li>
   </ul>
 </li>
 <li>
   <a href="#インストール">インストール</a>
   <ul>
     <li><a href="#PyPI">PyPI</a></li>
     <li><a href="#ソースからインストール">ソースからインストール</a></li>
   </ul>
 </li>
 <li><a href="#Docker-の使用">Docker の使用</a></li>
 <li><a href="#コミュニティ">コミュニティ</a></li>
 <li><a href="#コントリビュート">コントリビュート</a></li>
 <li><a href="#引用">引用</a></li>
</ul>

## なぜ Colossal-AI なのか
<div align="center">
   <a href="https://youtu.be/KnXSfjqkKN0">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/JamesDemmel_Colossal-AI.png" width="600" />
   </a>

   James Demmel 教授 (UC Berkeley): Colossal-AI は、AI モデルのトレーニングを効率的、簡単、スケーラブルにします。
</div>

<p align="right">(<a href="#top">トップへ戻る</a>)</p>

## 特徴

Colossal-AI は、あなたのために並列コンポーネントのコレクションを提供します。
私たちは、ノートパソコン上でモデルを書くのと同じように、分散ディープラーニングモデルを書くことをサポートすることを目指しています。
数行で分散学習と推論を開始できる、ユーザーフレンドリーなツールを提供します。

- 並列化戦略
  - データ並列
  - パイプライン並列
  - 1D, [2D](https://arxiv.org/abs/2104.05343), [2.5D](https://arxiv.org/abs/2105.14500), [3D](https://arxiv.org/abs/2105.14450) テンソル並列
  - [シーケンス並列](https://arxiv.org/abs/2105.13120)
  - [ゼロ冗長オプティマイザ (ZeRO)](https://arxiv.org/abs/1910.02054)
  - [自動並列m](https://arxiv.org/abs/2302.02599)

- 異種メモリ管理
  - [PatrickStar](https://arxiv.org/abs/2108.05818)

- フレンドリーな使い方
  - 設定ファイルに基づく並列性

<p align="right">(<a href="#top">トップへ戻る</a>)</p>

## 現実世界における Colossal-AI

### Colossal-LLaMA-2

- 7B: 数百ドルを使った半日のトレーニングで、主流の大規模モデル、オープンソース、商用フリーのドメイン特化型 LLM ソリューションと同様の結果が得られます。
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA-2)
[[ブログ]](https://www.hpc-ai.tech/blog/one-half-day-of-training-using-a-few-hundred-dollars-yields-similar-results-to-mainstream-large-models-open-source-and-commercial-free-domain-specific-llm-solution)
[[HuggingFace モデルウェイト]](https://huggingface.co/hpcai-tech/Colossal-LLaMA-2-7b-base)
[[Modelscope モデルウェイト]](https://www.modelscope.cn/models/colossalai/Colossal-LLaMA-2-7b-base/summary)

- 13B: わずか 5000 ドルで、洗練された 13B のプライベートモデルを建設。
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA-2)
[[ブログ]](https://hpc-ai.com/blog/colossal-llama-2-13b)
[[HuggingFace モデルウェイト]](https://huggingface.co/hpcai-tech/Colossal-LLaMA-2-13b-base)
[[Modelscope モデルウェイト]](https://www.modelscope.cn/models/colossalai/Colossal-LLaMA-2-13b-base/summary)

|              Model              |  Backbone  | Tokens Consumed |     MMLU (5-shot)    | CMMLU (5-shot)| AGIEval (5-shot) | GAOKAO (0-shot) | CEval (5-shot)  |
| :-----------------------------: | :--------: | :-------------: | :------------------: | :-----------: | :--------------: | :-------------: | :-------------: |
|          Baichuan-7B            |     -      |      1.2T       |    42.32 (42.30)     | 44.53 (44.02) |        38.72     |       36.74     |       42.80     |
|       Baichuan-13B-Base         |     -      |      1.4T       |    50.51 (51.60)     | 55.73 (55.30) |        47.20     |       51.41     |       53.60     |
|       Baichuan2-7B-Base         |     -      |      2.6T       |    46.97 (54.16)     | 57.67 (57.07) |        45.76     |       52.60     |       54.00     |
|       Baichuan2-13B-Base        |     -      |      2.6T       |    54.84 (59.17)     | 62.62 (61.97) |        52.08     |       58.25     |       58.10     |
|           ChatGLM-6B            |     -      |      1.0T       |    39.67 (40.63)     |   41.17 (-)   |        40.10     |       36.53     |       38.90     |
|          ChatGLM2-6B            |     -      |      1.4T       |    44.74 (45.46)     |   49.40 (-)   |        46.36     |       45.49     |       51.70     |
|          InternLM-7B            |     -      |      1.6T       |    46.70 (51.00)     |   52.00 (-)   |        44.77     |       61.64     |       52.80     |
|            Qwen-7B              |     -      |      2.2T       |    54.29 (56.70)     | 56.03 (58.80) |        52.47     |       56.42     |       59.60     |
|           Llama-2-7B            |     -      |      2.0T       |    44.47 (45.30)     |   32.97 (-)   |        32.60     |       25.46     |         -       |
| Linly-AI/Chinese-LLaMA-2-7B-hf  | Llama-2-7B |      1.0T       |        37.43         |     29.92     |        32.00     |       27.57     |         -       |
| wenge-research/yayi-7b-llama2   | Llama-2-7B |        -        |        38.56         |     31.52     |        30.99     |       25.95     |         -       |
| ziqingyang/chinese-llama-2-7b   | Llama-2-7B |        -        |        33.86         |     34.69     |        34.52     |       25.18     |        34.2     |
| TigerResearch/tigerbot-7b-base  | Llama-2-7B |      0.3T       |        43.73         |     42.04     |        37.64     |       30.61     |         -       |
|  LinkSoul/Chinese-Llama-2-7b    | Llama-2-7B |        -        |        48.41         |     38.31     |        38.45     |       27.72     |         -       |
|       FlagAlpha/Atom-7B         | Llama-2-7B |      0.1T       |        49.96         |     41.10     |        39.83     |       33.00     |         -       |
| IDEA-CCNL/Ziya-LLaMA-13B-v1.1   | Llama-13B  |      0.11T      |        50.25         |     40.99     |        40.04     |       30.54     |         -       |
|  **Colossal-LLaMA-2-7b-base**   | Llama-2-7B |   **0.0085T**   |        53.06         |     49.89     |        51.48     |       58.82     |        50.2     |
|  **Colossal-LLaMA-2-13b-base**  | Llama-2-13B |   **0.025T**    |        56.42         |     61.80     |        54.69     |       69.53     |        60.3     |


### ColossalChat

<div align="center">
   <a href="https://www.youtube.com/watch?v=HcTiHzApHm0">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/ColossalChat%20YouTube.png" width="700" />
   </a>
</div>

[ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat): [ChatGPT](https://openai.com/blog/chatgpt/) をクローンするためのオープンソースのソリューションで、完全な RLHF パイプラインを備えています。
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)
[[ブログ]](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)
[[デモ]](https://www.youtube.com/watch?v=HcTiHzApHm0)
[[チュートリアル]](https://www.youtube.com/watch?v=-qFBZFmOJfg)

<p id="ColossalChat-Speed" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/ColossalChat%20Speed.jpg" width=450/>
</p>

- RLHF PPO Stage3 トレーニングが最大 10 倍高速化

<p id="ColossalChat_scaling" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/ChatGPT%20scaling.png" width=800/>
</p>

- シングルサーバーのトレーニングで最大 7.73 倍、シングル GPU の推論で 1.42 倍高速化

<p id="ColossalChat-1GPU" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/ChatGPT-1GPU.jpg" width=450/>
</p>

- 1 つの GPU でモデル容量が最大 10.3 倍増加
- ミニデモのトレーニング処理に必要な GPU メモリはわずか 1.62GB（コンシューマーグレードの GPU であれば何でも可）

<p id="ColossalChat-LoRA" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/LoRA%20data.jpg" width=600/>
</p>

- シングル GPU でファインチューニングモデルの能力を最大 3.7 倍向上
- 十分に高い実行速度を維持

<p align="right">(<a href="#top">トップへ戻る</a>)</p>


### AIGC
[Stable Diffusion v1](https://github.com/CompVis/stable-diffusion) や [Stable Diffusion v2](https://github.com/Stability-AI/stablediffusion) などの AIGC（AI 生成コンテンツ）モデルの高速化。
<p id="diffusion_train" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Stable%20Diffusion%20v2.png" width=800/>
</p>

- [トレーニング](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion): Stable Diffusion のメモリ消費量を最大 5.6 倍、ハードウェアコストを最大46倍削減（A100 から RTX3060 へ）。

<p id="diffusion_demo" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/DreamBooth.png" width=800/>
</p>

- [DreamBooth ファインチューニング](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/dreambooth): 希望する被写体の画像を 3～5 枚使用するだけで、モデルをパーソナライズできます。

<p id="inference-sd" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Stable%20Diffusion%20Inference.jpg" width=800/>
</p>

- [推論](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion): 推論 GPU のメモリ消費量を 2.5 倍に削減。


<p align="right">(<a href="#top">トップへ戻る</a>)</p>

### Biomedicine
[AlphaFold Protein Structure](https://alphafold.ebi.ac.uk/) の加速

<p id="FastFold" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/FastFold.jpg" width=800/>
</p>

- [FastFold](https://github.com/hpcaitech/FastFold): GPU クラスタ上での学習と推論の高速化、データ処理の高速化、10000 残基以上の配列を含む推論。

<p id="FastFold-Intel" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/data%20preprocessing%20with%20Intel.jpg" width=600/>
</p>

- [FastFold with Intel](https://github.com/hpcaitech/FastFold): 3 倍の推論加速と 39％ のコスト削減。

<p id="xTrimoMultimer" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/xTrimoMultimer_Table.jpg" width=800/>
</p>

- [xTrimoMultimer](https://github.com/biomap-research/xTrimoMultimer): タンパク質のモノマーとマルチマーの構造予測を 11 倍高速化。


<p align="right">(<a href="#top">トップへ戻る</a>)</p>

## パラレルトレーニングデモ
### LLaMA2
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/llama2_pretraining.png" width=600/>
</p>

- 700 億パラメータの LLaMA2 モデル学習が 195% 高速化
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/llama2)
[[ブログ]](https://www.hpc-ai.tech/blog/70b-llama2-training)

### LLaMA1
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/images/LLaMA_pretraining.png" width=600/>
</p>

- 650 億パラメータの大規模モデルのプリトレーニングを 38% 高速化
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/example/llama/examples/language/llama)
[[ブログ]](https://www.hpc-ai.tech/blog/large-model-pretraining)

### MoE
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/images/MOE_training.png" width=800/>
</p>

- MoE の並列性が強化され、オープンソースの MoE モデルトレーニングは 9 倍効率的
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/openmoe)
[[ブログ]](https://www.hpc-ai.tech/blog/enhanced-moe-parallelism-open-source-moe-model-training-can-be-9-times-more-efficient)

### GPT-3
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT3-v5.png" width=700/>
</p>

- GPU リソースを 50% 節約し、10.7% の高速化

### GPT-2
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2.png" width=800/>

- GPU のメモリ消費量を 11 倍削減、テンソル並列処理で超線形スケーリング効率を実現

<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/(updated)GPT-2.png" width=800>

- 同じハードウェアで24倍のモデルサイズ
- 3倍以上の加速
### BERT
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BERT.png" width=800/>

- 2倍速いトレーニング、または 50％ 長いシークエンス長

### PaLM
- [PaLM-colossalai](https://github.com/hpcaitech/PaLM-colossalai): Google のパスウェイ言語モデル([PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html))のスケーラブルな実装。

### OPT
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/OPT_update.png" width=800/>

- [Open Pretrained Transformer (OPT)](https://github.com/facebookresearch/metaseq)は、Meta 社が公開した 1750 億パラメータの AI 言語モデルであり、事前に学習されたモデルの重みが公開されているため、AI プログラマーは様々な下流タスクやアプリケーションのデプロイを行うことができる。
- 回線コストを抑えて OPT を 45% 高速微調整。[[例]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/opt) [[オンライン給仕]](https://colossalai.org/docs/advanced_tutorials/opt_service)

詳しくは[ドキュメント](https://www.colossalai.org/)と[サンプル](https://github.com/hpcaitech/ColossalAI/tree/main/examples)をご覧ください。

### ViT
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/ViT.png" width="450" />
</p>

- バッチサイズが 14 倍、テンソル並列度が 64 の場合、トレーニングが5倍速くなる。

### 推薦システムモデル
- [Cached Embedding](https://github.com/hpcaitech/CachedEmbedding)は、ソフトウェアキャッシュを利用することで、より少ない GPU メモリ予算でより大きなエンベッディングテーブルを学習することができます。

<p align="right">(<a href="#top">トップへ戻る</a>)</p>

## シングル GPU トレーニングデモ

### GPT-2
<p id="GPT-2-Single" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2-GPU1.png" width=450/>
</p>

- 同じハードウェアで 20 倍のモデルサイズ

<p id="GPT-2-NVME" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2-NVME.png" width=800/>
</p>

- 同じハードウェアで 120 倍のモデルサイズ（RTX 3080）

### PaLM
<p id="PaLM-Single" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/PaLM-GPU1.png" width=450/>
</p>

- 同じハードウェアで 34 倍のモデルサイズ

<p align="right">(<a href="#top">トップへ戻る</a>)</p>


## 推論
<p id="SwiftInfer" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/SwiftInfer.jpg" width=800/>
</p>

- [SwiftInfer](https://github.com/hpcaitech/SwiftInfer): 推論性能が 46% 向上、オープンソースソリューションが多ラウンド会話における LLM の長さ制限を突破

<p id="GPT-3-Inference" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference_GPT-3.jpg" width=800/>
</p>

- [Energon-AI](https://github.com/hpcaitech/EnergonAI): 同じハードウェアで 50％ の推論加速

<p id="OPT-Serving" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BLOOM%20serving.png" width=600/>
</p>

- [OPT Serving](https://colossalai.org/docs/advanced_tutorials/opt_service): 1,750 億パラメータの OPT オンラインサービスを試

<p id="BLOOM-Inference" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BLOOM%20Inference.PNG" width=800/>
</p>

- [BLOOM](https://github.com/hpcaitech/EnergonAI/tree/main/examples/bloom): 1,760 億パラメータ BLOOM のハードウェア導入コストを 10 倍以上削減。

<p align="right">(<a href="#top">トップへ戻る</a>)</p>

## インストール

必要条件:
- PyTorch >= 1.11 及び PyTorch <= 2.1
- Python >= 3.7
- CUDA >= 11.0
- [NVIDIA GPU コンピューティング能力](https://developer.nvidia.com/cuda-gpus) >= 7.0 (V100/RTX20 以上)
- Linux OS

インストールで何か問題が発生した場合は、このリポジトリで [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose) を提起してください。

### PyPI からインストール

Colossal-AI は以下のコマンドで簡単にインストールできます。**デフォルトでは、インストール時に PyTorch の拡張機能はビルドされません。**

```bash
pip install colossalai
```

**注: 今のところ Linux のみがサポートされています。**

しかし、インストール中に PyTorch の拡張機能をビルドしたい場合は、 `CUDA_EXT=1` を設定します。

```bash
CUDA_EXT=1 pip install colossalai
```

**そうでなければ、CUDAカーネルは実際に必要なときに実行時にビルドされることになる。**

また、毎週ナイトリーバージョンを PyPI にリリースしています。これにより、メインブランチの未リリースの機能やバグ修正にアクセスできるようになります。
インストールは以下の方法

```bash
pip install colossalai-nightly
```

### ソースからダウンロード

> Colossal-AI のバージョンはリポジトリのメインブランチに合わせます。何か問題が発生したら、遠慮なく issue を送信してください。 :)

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# colossalai のインストール
pip install .
```

デフォルトでは、CUDA/C++ カーネルはコンパイルされません。ColossalAI は実行時にそれらをビルドします。
CUDA カーネルフュージョンをインストールして有効にする場合（フューズドオプティマイザを使用する場合、インストールは必須です）:

```shell
CUDA_EXT=1 pip install .
```

CUDA 10.2 を使っているユーザーは、ソースから ColossalAI をビルドすることができます。ただし、cub ライブラリを手動でダウンロードし、対応するディレクトリにコピーする必要があります。

```bash
# リポジトリをクローン
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# cub ライブラリをダウンロード
wget https://github.com/NVIDIA/cub/archive/refs/tags/1.8.0.zip
unzip 1.8.0.zip
cp -r cub-1.8.0/cub/ colossalai/kernel/cuda_native/csrc/kernels/include/

# インストール
CUDA_EXT=1 pip install .
```

<p align="right">(<a href="#top">トップへ戻る</a>)</p>

## Docker の使用

### DockerHub からのプル

私たちの [DockerHub ページ](https://hub.docker.com/r/hpcaitech/colossalai)から直接 Docker イメージをプルすることができます。イメージはリリース時に自動的にアップロードされます。


### 自分でビルドする

以下のコマンドを実行し、提供された Dockerfile から Docker イメージをビルドする。

> Colossal-AI をゼロからビルドするには GPU サポートが必要で、`docker build` を実行する際に Nvidia Docker Runtime をデフォルトとして使用する必要があります。詳細は[こちら](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime)を参照にして下さい。
> 私たちの[プロジェクトページ](https://www.colossalai.org)から直接 Colossal-AI をインストールすることをお勧めします。


```bash
cd ColossalAI
docker build -t colossalai ./docker
```

以下のコマンドを実行して、対話モードで docker コンテナを起動する。

```bash
docker run -ti --gpus all --rm --ipc=host colossalai bash
```

<p align="right">(<a href="#top">トップへ戻る</a>)</p>

## コミュニティ

[フォーラム](https://github.com/hpcaitech/ColossalAI/discussions)、[Slack](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w)、[WeChat (微信)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png "qrcode") の Colossal-AI コミュニティに参加し、提案、フィードバック、質問をエンジニアリングチームと共有しましょう。

## コントリビュート
[BLOOM](https://bigscience.huggingface.co/) や [Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion) の成功例を参考に、計算能力、データセット、モデルを持つ開発者やパートナーは誰でも、Colossal-AI コミュニティに参加し、大 AI モデルの時代に向けて努力することを歓迎します！

以下の方法でご連絡またはご参加いただけます:
1. [Leaving a Star ⭐](https://github.com/hpcaitech/ColossalAI/stargazers) をクリックし、「いいね！」や「応援しています！」の意思表示をしてください。ありがとう！
2. [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose) を投稿したり、GitHub に PR を投稿したりする場合は、[Contributing](https://github.com/hpcaitech/ColossalAI/blob/main/CONTRIBUTING.md) のガイドラインに従ってください
3. 正式な提案書をEメール contact@hpcaitech.com までお送りください。

素晴らしいコントリビューターの皆さん、本当にありがとう！

<a href="https://github.com/hpcaitech/ColossalAI/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hpcaitech/ColossalAI"  width="800px"/>
</a>


<p align="right">(<a href="#top">トップへ戻る</a>)</p>


## CI/CD

私たちは、[GitHub Actions](https://github.com/features/actions) の力を借りて、開発、リリース、デプロイのワークフローを自動化しています。自動化されたワークフローがどのように運用されているかについては、こちらの[ドキュメント](../.github/workflows/README.md)をご覧ください。


## 引用

このプロジェクトは、いくつかの関連プロジェクト（我々のチームによるものもあれば、他の組織によるものもある）に触発されている。[参考文献リスト](./REFERENCE.md)に記載されているように、これらの素晴らしいプロジェクトに謝意を表したいと思います。

このプロジェクトを引用するには、以下のBibTeX引用を使用できます。

```
@inproceedings{10.1145/3605573.3605613,
author = {Li, Shenggui and Liu, Hongxin and Bian, Zhengda and Fang, Jiarui and Huang, Haichen and Liu, Yuliang and Wang, Boxiang and You, Yang},
title = {Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
year = {2023},
isbn = {9798400708435},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3605573.3605613},
doi = {10.1145/3605573.3605613},
abstract = {The success of Transformer models has pushed the deep learning model scale to billions of parameters, but the memory limitation of a single GPU has led to an urgent need for training on multi-GPU clusters. However, the best practice for choosing the optimal parallel strategy is still lacking, as it requires domain expertise in both deep learning and parallel computing. The Colossal-AI system addressed the above challenge by introducing a unified interface to scale your sequential code of model training to distributed environments. It supports parallel training methods such as data, pipeline, tensor, and sequence parallelism and is integrated with heterogeneous training and zero redundancy optimizer. Compared to the baseline system, Colossal-AI can achieve up to 2.76 times training speedup on large-scale models.},
booktitle = {Proceedings of the 52nd International Conference on Parallel Processing},
pages = {766–775},
numpages = {10},
keywords = {datasets, gaze detection, text tagging, neural networks},
location = {Salt Lake City, UT, USA},
series = {ICPP '23}
}
```

Colossal-AI は、トップカンファレンス [NeurIPS](https://nips.cc/)、[SC](https://sc22.supercomputing.org/)、[AAAI](https://aaai.org/Conferences/AAAI-23/)、[PPoPP](https://ppopp23.sigplan.org/)、[CVPR](https://cvpr2023.thecvf.com/)、[ISC](https://www.isc-hpc.com/)、[NVIDIA GTC](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-S51482/) などで公式チュートリアルとして採用されています。

<p align="right">(<a href="#top">トップへ戻る</a>)</p>
