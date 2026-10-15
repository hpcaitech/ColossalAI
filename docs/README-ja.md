# Colossal-AI
<div id="top" align="center">

   [![logo](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/colossal-ai_logo_vertical.png)](https://www.colossalai.org/)

   Colossal-AI: 大規模AIモデルをより安く、より速く、よりアクセスしやすく

   <h3> <a href="https://arxiv.org/abs/2110.14883"> 論文 </a> |
   <a href="https://www.colossalai.org/"> ドキュメント </a> |
   <a href="https://github.com/hpcaitech/ColossalAI/tree/main/examples"> サンプル </a> |
   <a href="https://github.com/hpcaitech/ColossalAI/discussions"> フォーラム </a> |
   <a href="https://colossalai.org/zh-Hans/docs/get_started/bonus/">GPUクラウドプレイグラウンド </a> |
   <a href="https://hpc-ai.com/blog"> ブログ </a></h3>

   [![GitHub Repo stars](https://img.shields.io/github/stars/hpcaitech/ColossalAI?style=social)](https://github.com/hpcaitech/ColossalAI/stargazers)
   [![Build](https://github.com/hpcaitech/ColossalAI/actions/workflows/build_on_schedule.yml/badge.svg)](https://github.com/hpcaitech/ColossalAI/actions/workflows/build_on_schedule.yml)
   [![Documentation](https://readthedocs.org/projects/colossalai/badge/?version=latest)](https://colossalai.readthedocs.io/en/latest/?badge=latest)
   [![CodeFactor](https://www.codefactor.io/repository/github/hpcaitech/colossalai/badge)](https://www.codefactor.io/repository/github/hpcaitech/colossalai)
   [![HuggingFace badge](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Join-yellow)](https://huggingface.co/hpcai-tech)
   [![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://github.com/hpcaitech/public_assets/tree/main/colossalai/contact/slack)
   [![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png)


   | [English](../README.md) | [中文](README-zh-Hans.md) | [日本語](README-ja.md) |

</div>

## エンタープライズグレードGPUでColossal-AIを即座に実行

セットアップは不要です。[**HPC-AI Cloud**](https://hpc-ai.com/?utm_source=github&utm_medium=social&utm_campaign=promotion-colossalai)上で事前設定済みの強力なColossal-AI環境にアクセスできます。

ワンクリックでモデルのトレーニングとAIワークロードのスケーリングが可能です！

* **NVIDIA Blackwell B200s**: 次世代AIパフォーマンスを体験（[ベンチマークを見る](https://hpc-ai.com/blog/b200)）。クラウドで**$2.47/時間**から利用可能。
* **コスト効率の高いH200クラスター**: オンデマンドレンタルでわずか**$1.99/時間**からプレミアパフォーマンスを実現。

[**今すぐ始めて無料クレジットを獲得 →**](https://hpc-ai.com/?utm_source=github&utm_medium=social&utm_campaign=promotion-colossalai)

<div align="center">
   <a href="https://hpc-ai.com/?utm_source=github&utm_medium=social&utm_campaign=promotion-colossalai">
   <img src="https://github.com/hpcaitech/public_assets/blob/main/colossalai/img/2-3.png" width="850" />
   </a>
</div>

## 半額でトップオープンモデルに即座にアクセス

面倒な設定は不要。[**HPC-AI Model APIs**](https://hpc-ai.com/model-apis?utm_source=github&utm_medium=social&utm_campaign=promotion-colossalai)を通じて、強力なロングコンテキストLLMにシームレスにアクセスできます。

HPC-AI Model APIsでAIエージェント、チャットボット、RAGアプリケーションを構築しましょう！

* **最新・最高性能モデル**: Kimi 2.5、MiniMax 2.5、GLM 5.1で最先端のパフォーマンスを体験。200万以上のコンテキストウィンドウや複雑なコーディングタスクに最適。

* **圧倒的な価格**: APIエンドポイントに過剰な費用を払う必要はありません。OpenRouterより最大50%安い価格でプレミア推論速度を実現。

[**今すぐ始めて$4の無料クレジットを獲得 →**](https://www.hpc-ai.com/account/signup?redirectUrl=/models-console/models&invitation_code=HPCAI-MAPI&utm_source=google&utm_medium=social&utm_id=newlaunch)

<div align="center">
   <a href="https://hpc-ai.com/model-apis?utm_source=github&utm_medium=social&utm_campaign=promotion-colossalai">
   <img src="https://github.com/hpcaitech/public_assets/blob/main/colossalai/img/model%20APIs.png" width="850" />
   </a>
</div>

### Colossal-AIベンチマーク

これらのパフォーマンス向上が実世界のアプリケーションにどのように反映されるかを確認するため、Colossal-AIを使用してLLaMA系モデルの大規模言語モデルトレーニングベンチマークを実施しました。テストは7Bおよび70Bモデルに対して、それぞれ8枚および16枚のGPU構成で実行されました。

|              GPU              |  GPUs  | モデルサイズ |    並列化方式    | DPあたりのバッチサイズ | シーケンス長 | スループット | TFLOPS/GPU  | ピークメモリ(MiB)  |
| :-----------------------------: | :--------: | :-------------: | :------------------: | :-----------: | :--------------: | :-------------: | :-------------: | :-------------: |
|         H200            |     8     |      7B       |   zero2(dp8)     | 36 |        4096     |       17.13 samp/s     |       534.18     |       119040.02     |
|         H200            |     16     |      70B       |   zero2     | 48 |        4096     |       3.27 samp/s     |       469.1     |       150032.23     |
|         B200            |     8     |      7B       |   zero1(dp2)+tp2+pp4     | 128 |        4096     |       25.83 samp/s     |       805.69     |       100119.77     |
|         H200            |     16     |      70B       |   zero1(dp2)+tp2+pp4     | 128 |        4096     |       5.66 samp/s     |       811.79     |       100072.02     |

Colossal-AIベンチマークの結果は最も実用的な知見を提供します。8枚のGPUでの7Bモデルでは、**B200は50%高いスループット**を達成し、GPU当たりのTFLOPSも大幅に向上しました。16枚のGPUでの70Bモデルでも、B200は明確な優位性を示し、**スループットとGPU当たりのTFLOPSが70%以上向上**しました。これらの数値は、B200のパフォーマンス向上が大規模モデルのトレーニング時間の短縮に直接つながることを示しています。

## 最新ニュース
* [2025/02] [DeepSeek 671Bファインチューニングガイド公開 — ワンクリックでアップグレード版DeepSeekスイートを解放、AIプレイヤー歓喜！](https://company.hpc-ai.com/blog/shocking-release-deepseek-671b-fine-tuning-guide-revealed-unlock-the-upgraded-deepseek-suite-with-one-click-ai-players-ecstatic)
* [2024/12] [動画生成モデルの開発コストが50%削減！H200 GPUバウチャー付きのオープンソースソリューションが利用可能に](https://company.hpc-ai.com/blog/the-development-cost-of-video-generation-models-has-saved-by-50-open-source-solutions-are-now-available-with-h200-gpu-vouchers) [[コード]](https://github.com/hpcaitech/Open-Sora/blob/main/scripts/train.py) [[バウチャー]](https://colossalai.org/zh-Hans/docs/get_started/bonus/)
* [2024/10] [低コストなSoraライクアプリの構築方法は？あなたのためのソリューション](https://company.hpc-ai.com/blog/how-to-build-a-low-cost-sora-like-app-solutions-for-you)
* [2024/09] [シンガポールのスタートアップHPC-AI Tech、動画生成AIモデルとGPUプラットフォーム構築のためシリーズAで5000万ドルを調達](https://company.hpc-ai.com/blog/singapore-startup-hpc-ai-tech-secures-50-million-usd-in-series-a-funding-to-build-the-video-generation-ai-model-and-gpu-platform)
* [2024/09] [FP8混合精度トレーニングのアップグレードによりAI大規模モデルのトレーニングコストを30%削減、必要なのはたった1行のコード](https://company.hpc-ai.com/blog/reducing-ai-large-model-training-costs-by-30-requires-just-a-single-line-of-code-from-fp8-mixed-precision-training-upgrades)
* [2024/06] [Open-Soraがオープンソースを継続：ワンクリックで任意の16秒720p HDビデオを生成、モデルウェイトはすぐに使用可能](https://hpc-ai.com/blog/open-sora-from-hpc-ai-tech-team-continues-open-source-generate-any-16-second-720p-hd-video-with-one-click-model-weights-ready-to-use)
* [2024/05] [大規模AIモデルの推論速度が2倍に、Colossal-Inferenceオープンソースリリース](https://hpc-ai.com/blog/colossal-inference)
* [2024/04] [Open-Soraが大幅アップグレード：オープンソースで16秒動画生成と720p解像度に対応](https://hpc-ai.com/blog/open-soras-comprehensive-upgrade-unveiled-embracing-16-second-video-generation-and-720p-resolution-in-open-source)
* [2024/04] [LLaMA3シリーズに最適化された推論・ファインチューニング・事前学習の最もコスト効率の高いソリューション](https://hpc-ai.com/blog/most-cost-effective-solutions-for-inference-fine-tuning-and-pretraining-tailored-to-llama3-series)

## 目次
<ul>
 <li><a href="#なぜColossal-AIなのか">なぜColossal-AIなのか</a> </li>
 <li><a href="#特徴">特徴</a> </li>
 <li>
   <a href="#実世界でのColossal-AI">実世界でのColossal-AI</a>
   <ul>
     <li><a href="#Open-Sora">Open-Sora: Soraライク動画生成モデルの完全なモデルパラメータ、トレーニング詳細、その他すべてを公開</a></li>
     <li><a href="#Colossal-LLaMA-2">Colossal-LLaMA-2: 数百ドルで半日のトレーニングにより主要な大規模モデルと同等の結果を実現、オープンソース・商用フリーのドメイン特化型LLMソリューション</a></li>
     <li><a href="#ColossalChat">ColossalChat: 完全なRLHFパイプラインによるChatGPTクローンのオープンソースソリューション</a></li>
     <li><a href="#AIGC">AIGC: Stable Diffusionの高速化</a></li>
     <li><a href="#生体医学">生体医学: AlphaFoldタンパク質構造の高速化</a></li>
   </ul>
 </li>
 <li>
   <a href="#並列トレーニングデモ">並列トレーニングデモ</a>
   <ul>
     <li><a href="#LLaMA3">LLaMA 1/2/3 </a></li>
     <li><a href="#MoE">MoE</a></li>
     <li><a href="#GPT-3">GPT-3</a></li>
     <li><a href="#GPT-2">GPT-2</a></li>
     <li><a href="#BERT">BERT</a></li>
     <li><a href="#PaLM">PaLM</a></li>
     <li><a href="#OPT">OPT</a></li>
     <li><a href="#ViT">ViT</a></li>
     <li><a href="#レコメンドシステムモデル">レコメンドシステムモデル</a></li>
   </ul>
 </li>
 <li>
   <a href="#シングルGPUトレーニングデモ">シングルGPUトレーニングデモ</a>
   <ul>
     <li><a href="#GPT-2-Single">GPT-2</a></li>
     <li><a href="#PaLM-Single">PaLM</a></li>
   </ul>
 </li>
 <li>
   <a href="#推論">推論</a>
   <ul>
     <li><a href="#Colossal-Inference">Colossal-Inference: 大規模AIモデルの推論速度が2倍に</a></li>
     <li><a href="#Grok-1">Grok-1: 314Bモデルの PyTorch + HuggingFace推論</a></li>
     <li><a href="#SwiftInfer">SwiftInfer: マルチラウンド会話におけるLLMの長さ制限を突破、46%高速化</a></li>
   </ul>
 </li>
 <li>
   <a href="#インストール">インストール</a>
   <ul>
     <li><a href="#PyPI">PyPI</a></li>
     <li><a href="#ソースからインストール">ソースからインストール</a></li>
   </ul>
 </li>
 <li><a href="#Dockerの使用">Dockerの使用</a></li>
 <li><a href="#コミュニティ">コミュニティ</a></li>
 <li><a href="#コントリビューション">コントリビューション</a></li>
 <li><a href="#引用">引用</a></li>
</ul>

## なぜColossal-AIなのか
<div align="center">
   <a href="https://youtu.be/KnXSfjqkKN0">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/JamesDemmel_Colossal-AI.png" width="600" />
   </a>

   James Demmel教授（カリフォルニア大学バークレー校）: Colossal-AIはAIモデルのトレーニングを効率的、簡単、かつスケーラブルにします。
</div>

<p align="right">(<a href="#top">トップに戻る</a>)</p>

## 特徴

Colossal-AIは並列コンポーネントのコレクションを提供します。ノートパソコンでモデルを書くのと同じように、分散型ディープラーニングモデルを記述できることを目指しています。数行のコードで分散トレーニングと推論を開始するための使いやすいツールを提供します。

- 並列化戦略
  - データ並列
  - パイプライン並列
  - 1D、[2D](https://arxiv.org/abs/2104.05343)、[2.5D](https://arxiv.org/abs/2105.14500)、[3D](https://arxiv.org/abs/2105.14450)テンソル並列
  - [シーケンス並列](https://arxiv.org/abs/2105.13120)
  - [ゼロ冗長オプティマイザー (ZeRO)](https://arxiv.org/abs/1910.02054)
  - [自動並列化](https://arxiv.org/abs/2302.02599)

- ヘテロジニアスメモリ管理
  - [PatrickStar](https://arxiv.org/abs/2108.05818)

- 使いやすさ
  - 設定ファイルによる並列化

<p align="right">(<a href="#top">トップに戻る</a>)</p>

## 実世界でのColossal-AI
### Open-Sora

[Open-Sora](https://github.com/hpcaitech/Open-Sora)：Soraライク動画生成モデルの完全なモデルパラメータ、トレーニング詳細、その他すべてを公開
[[コード]](https://github.com/hpcaitech/Open-Sora)
[[ブログ]](https://hpc-ai.com/blog/open-sora-from-hpc-ai-tech-team-continues-open-source-generate-any-16-second-720p-hd-video-with-one-click-model-weights-ready-to-use)
[[モデルウェイト]](https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#model-weights)
[[デモ]](https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#-latest-demo)
[[GPUクラウドプレイグラウンド]](https://cloud.luchentech.com/)
[[OpenSoraイメージ]](https://cloud.luchentech.com/doc/docs/image/open-sora/)

<div align="center">
   <a href="https://youtu.be/ilMQpU71ddI?si=J4JSPzZ03ycYmlki">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/sora/opensora-v1.2.png" width="700" />
   </a>
</div>

<p align="right">(<a href="#top">トップに戻る</a>)</p>

### Colossal-LLaMA-2

[[GPUクラウドプレイグラウンド]](https://cloud.luchentech.com/)
[[LLaMA3イメージ]](https://cloud.luchentech.com/doc/docs/image/llama)

- 7B: 数百ドルで半日のトレーニングにより主要な大規模モデルと同等の結果を実現、オープンソース・商用フリーのドメイン特化型LLMソリューション。
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA-2)
[[ブログ]](https://www.hpc-ai.tech/blog/one-half-day-of-training-using-a-few-hundred-dollars-yields-similar-results-to-mainstream-large-models-open-source-and-commercial-free-domain-specific-llm-solution)
[[HuggingFaceモデルウェイト]](https://huggingface.co/hpcai-tech/Colossal-LLaMA-2-7b-base)
[[Modelscopeモデルウェイト]](https://www.modelscope.cn/models/colossalai/Colossal-LLaMA-2-7b-base/summary)

- 13B: わずか5000ドルで精緻な13Bプライベートモデルを構築。
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA-2)
[[ブログ]](https://hpc-ai.com/blog/colossal-llama-2-13b)
[[HuggingFaceモデルウェイト]](https://huggingface.co/hpcai-tech/Colossal-LLaMA-2-13b-base)
[[Modelscopeモデルウェイト]](https://www.modelscope.cn/models/colossalai/Colossal-LLaMA-2-13b-base/summary)

|              モデル              |  バックボーン  | 消費トークン数 |     MMLU (5-shot)    | CMMLU (5-shot)| AGIEval (5-shot) | GAOKAO (0-shot) | CEval (5-shot)  |
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

[ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat): 完全なRLHFパイプラインによる[ChatGPT](https://openai.com/blog/chatgpt/)クローンのオープンソースソリューション。
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)
[[ブログ]](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)
[[デモ]](https://www.youtube.com/watch?v=HcTiHzApHm0)
[[チュートリアル]](https://www.youtube.com/watch?v=-qFBZFmOJfg)

<p id="ColossalChat-Speed" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/ColossalChat%20Speed.jpg" width=450/>
</p>

- RLHF PPO Stage3トレーニングが最大10倍高速化

<p id="ColossalChat_scaling" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/ChatGPT%20scaling.png" width=800/>
</p>

- シングルサーバートレーニングで最大7.73倍、シングルGPU推論で1.42倍の高速化

<p id="ColossalChat-1GPU" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/ChatGPT-1GPU.jpg" width=450/>
</p>

- 1つのGPUでモデル容量が最大10.3倍に拡大
- ミニデモのトレーニングプロセスに必要なGPUメモリはわずか1.62GB（一般消費者向けGPUで対応可能）

<p id="ColossalChat-LoRA" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/LoRA%20data.jpg" width=600/>
</p>

- シングルGPUでファインチューニングモデルの容量を最大3.7倍に拡大
- 十分に高い実行速度を維持

<p align="right">(<a href="#top">トップに戻る</a>)</p>


### AIGC
[Stable Diffusion v1](https://github.com/CompVis/stable-diffusion)や[Stable Diffusion v2](https://github.com/Stability-AI/stablediffusion)などのAIGC（AI生成コンテンツ）モデルの高速化。
<p id="diffusion_train" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Stable%20Diffusion%20v2.png" width=800/>
</p>

- [トレーニング](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion): Stable Diffusionのメモリ消費を最大5.6倍削減、ハードウェアコストを最大46倍削減（A100からRTX3060へ）。

<p id="diffusion_demo" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/DreamBooth.png" width=800/>
</p>

- [DreamBoothファインチューニング](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/dreambooth): わずか3〜5枚の画像でモデルをパーソナライズ。

<p id="inference-sd" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Stable%20Diffusion%20Inference.jpg" width=800/>
</p>

- [推論](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion): 推論時のGPUメモリ消費を2.5倍削減。


<p align="right">(<a href="#top">トップに戻る</a>)</p>

### 生体医学
[AlphaFoldタンパク質構造](https://alphafold.ebi.ac.uk/)の高速化

<p id="FastFold" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/FastFold.jpg" width=800/>
</p>

- [FastFold](https://github.com/hpcaitech/FastFold): GPUクラスターでのトレーニングと推論を高速化、より高速なデータ処理、10000残基以上を含む推論シーケンスに対応。

<p id="FastFold-Intel" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/data%20preprocessing%20with%20Intel.jpg" width=600/>
</p>

- [FastFold with Intel](https://github.com/hpcaitech/FastFold): 推論を3倍高速化、コストを39%削減。

<p id="xTrimoMultimer" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/xTrimoMultimer_Table.jpg" width=800/>
</p>

- [xTrimoMultimer](https://github.com/biomap-research/xTrimoMultimer): タンパク質モノマーおよびマルチマーの構造予測を11倍高速化。


<p align="right">(<a href="#top">トップに戻る</a>)</p>

## 並列トレーニングデモ
### LLaMA3
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/images/LLaMA3-70B-H100.png" width=600/>
</p>

- 700億パラメータLLaMA3モデルのトレーニングを18%高速化
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/llama)
[[GPUクラウドプレイグラウンド]](https://cloud.luchentech.com/)
[[LLaMA3イメージ]](https://cloud.luchentech.com/doc/docs/image/llama)

### LLaMA2
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/llama2_pretraining.png" width=600/>
</p>

- 700億パラメータLLaMA2モデルのトレーニングを195%高速化
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/llama)
[[ブログ]](https://www.hpc-ai.tech/blog/70b-llama2-training)

### LLaMA1
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/images/LLaMA_pretraining.png" width=600/>
</p>

- 650億パラメータ大規模モデルの事前学習を38%高速化
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/llama)
[[ブログ]](https://www.hpc-ai.tech/blog/large-model-pretraining)

### MoE
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/images/MOE_training.png" width=800/>
</p>

- MoE並列化の強化、オープンソースMoEモデルのトレーニングが9倍効率化
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/openmoe)
[[ブログ]](https://www.hpc-ai.tech/blog/enhanced-moe-parallelism-open-source-moe-model-training-can-be-9-times-more-efficient)

### GPT-3
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT3-v5.png" width=700/>
</p>

- GPUリソースを50%節約、10.7%の高速化

### GPT-2
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2.png" width=800/>

- GPUメモリ消費を11倍削減、テンソル並列によるスーパーリニアなスケーリング効率

<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/(updated)GPT-2.png" width=800>

- 同じハードウェアで24倍大きなモデルサイズ
- 3倍以上の高速化
### BERT
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BERT.png" width=800/>

- トレーニングが2倍高速化、またはシーケンス長が50%拡大

### PaLM
- [PaLM-colossalai](https://github.com/hpcaitech/PaLM-colossalai): GoogleのPathways Language Model ([PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html))のスケーラブルな実装。

### OPT
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/OPT_update.png" width=800/>

- [Open Pretrained Transformer (OPT)](https://github.com/facebookresearch/metaseq)は、Metaがリリースした1750億パラメータのAI言語モデルで、事前学習済みモデルウェイトが公開されているため、AIプログラマーがさまざまな下流タスクやアプリケーションデプロイメントを実行するきっかけとなりました。
- 低コストでOPTのファインチューニングを45%高速化。[[サンプル]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/opt) [[オンラインサービング]](https://colossalai.org/docs/advanced_tutorials/opt_service)

詳細については[ドキュメント](https://www.colossalai.org/)と[サンプル](https://github.com/hpcaitech/ColossalAI/tree/main/examples)をご覧ください。

### ViT
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/ViT.png" width="450" />
</p>

- テンソル並列=64でバッチサイズが14倍、トレーニングが5倍高速化

### レコメンドシステムモデル
- [Cached Embedding](https://github.com/hpcaitech/CachedEmbedding): ソフトウェアキャッシュを活用し、より少ないGPUメモリ予算でより大きなエンベディングテーブルのトレーニングが可能。

<p align="right">(<a href="#top">トップに戻る</a>)</p>

## シングルGPUトレーニングデモ

### GPT-2
<p id="GPT-2-Single" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2-GPU1.png" width=450/>
</p>

- 同じハードウェアで20倍大きなモデルサイズ

<p id="GPT-2-NVME" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2-NVME.png" width=800/>
</p>

- 同じハードウェア（RTX 3080）で120倍大きなモデルサイズ

### PaLM
<p id="PaLM-Single" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/PaLM-GPU1.png" width=450/>
</p>

- 同じハードウェアで34倍大きなモデルサイズ

<p align="right">(<a href="#top">トップに戻る</a>)</p>


## 推論
### Colossal-Inference
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/colossal-inference-v1-1.png" width=1000/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/colossal-inference-v1-2.png" width=1000/>
</p>

 - 大規模AIモデルの推論速度が2倍に向上。一部のケースではvLLMのオフライン推論パフォーマンスと比較。
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/inference)
[[ブログ]](https://hpc-ai.com/blog/colossal-inference)
[[GPUクラウドプレイグラウンド]](https://cloud.luchentech.com/)
[[LLaMA3イメージ]](https://cloud.luchentech.com/doc/docs/image/llama)

### Grok-1
<p id="Grok-1" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/images/grok-1-inference.jpg" width=600/>
</p>

 - 3140億パラメータGrok-1の推論を3.8倍高速化。推論用のPython + PyTorch + HuggingFace版で使いやすい。

[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/grok-1)
[[ブログ]](https://hpc-ai.com/blog/314-billion-parameter-grok-1-inference-accelerated-by-3.8x-efficient-and-easy-to-use-pytorchhuggingface-version-is-here)
[[HuggingFace Grok-1 PyTorchモデルウェイト]](https://huggingface.co/hpcai-tech/grok-1)
[[ModelScope Grok-1 PyTorchモデルウェイト]](https://www.modelscope.cn/models/colossalai/grok-1-pytorch/summary)

### SwiftInfer
<p id="SwiftInfer" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/SwiftInfer.jpg" width=800/>
</p>

- [SwiftInfer](https://github.com/hpcaitech/SwiftInfer): 推論パフォーマンスが46%向上。マルチラウンド会話におけるLLMの長さ制限を突破するオープンソースソリューション。

<p align="right">(<a href="#top">トップに戻る</a>)</p>

## インストール

要件:
- PyTorch >= 2.2
- Python >= 3.7
- CUDA >= 11.0
- [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 7.0 (V100/RTX20以上)
- Linux OS

インストールに問題が発生した場合は、このリポジトリに[issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose)を作成してください。

### PyPIからインストール

以下のコマンドで簡単にColossal-AIをインストールできます。**デフォルトでは、インストール時にPyTorch拡張機能はビルドされません。**

```bash
pip install colossalai
```

**注意: 現在Linuxのみサポートしています。**

インストール時にPyTorch拡張機能をビルドしたい場合は、`BUILD_EXT=1`を設定してください。

```bash
BUILD_EXT=1 pip install colossalai
```

**それ以外の場合、CUDAカーネルは実際に必要なときにランタイムでビルドされます。**

また、毎週PyPIにナイトリーバージョンをリリースしています。これにより、メインブランチの未リリース機能やバグ修正にアクセスできます。
以下のコマンドでインストールできます。

```bash
pip install colossalai-nightly
```

### ソースからインストール

> Colossal-AIのバージョンはリポジトリのメインブランチと同期しています。問題が発生した場合はお気軽にissueを作成してください。:)

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# colossalaiをインストール
pip install .
```

デフォルトでは、CUDA/C++カーネルはコンパイルされません。ColossalAIはランタイムでビルドします。
CUDAカーネルフュージョンをインストールして有効にしたい場合（融合オプティマイザー使用時は必須）:

```shell
BUILD_EXT=1 pip install .
```

CUDA 10.2をお使いの方は、ソースからColossalAIをビルドできますが、cubライブラリを手動でダウンロードし、対応するディレクトリにコピーする必要があります。

```bash
# リポジトリをクローン
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# cubライブラリをダウンロード
wget https://github.com/NVIDIA/cub/archive/refs/tags/1.8.0.zip
unzip 1.8.0.zip
cp -r cub-1.8.0/cub/ colossalai/kernel/cuda_native/csrc/kernels/include/

# インストール
BUILD_EXT=1 pip install .
```

<p align="right">(<a href="#top">トップに戻る</a>)</p>

## Dockerの使用

### DockerHubからプル

[DockerHubページ](https://hub.docker.com/r/hpcaitech/colossalai)からDockerイメージを直接プルできます。イメージはリリース時に自動的にアップロードされます。


### 自分でビルド

提供されたDockerfileからDockerイメージをビルドするには、以下のコマンドを実行します。

> ゼロからColossal-AIをビルドするにはGPUサポートが必要です。`docker build`を実行する際にNvidia Docker Runtimeをデフォルトとして使用する必要があります。詳細は[こちら](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime)をご覧ください。
> [プロジェクトページ](https://www.colossalai.org)から直接Colossal-AIをインストールすることをお勧めします。


```bash
cd ColossalAI
docker build -t colossalai ./docker
```

以下のコマンドでインタラクティブモードでDockerコンテナを起動します。

```bash
docker run -ti --gpus all --rm --ipc=host colossalai bash
```

<p align="right">(<a href="#top">トップに戻る</a>)</p>

## コミュニティ

[フォーラム](https://github.com/hpcaitech/ColossalAI/discussions)、
[Slack](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w)、
[WeChat(微信)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png "qrcode")でColossal-AIコミュニティに参加し、エンジニアリングチームにご意見、フィードバック、質問を共有してください。

## コントリビューション
[BLOOM](https://bigscience.huggingface.co/)や[Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion)の成功事例を参考に、計算リソース、データセット、モデルをお持ちのすべての開発者やパートナーの皆さまを歓迎します。Colossal-AIコミュニティに参加し、大規模AIモデルの時代に向けて一緒に取り組みましょう！

以下の方法でお問い合わせまたはご参加いただけます：
1. [スターを付ける ⭐](https://github.com/hpcaitech/ColossalAI/stargazers) で応援とサポートを表明。ありがとうございます！
2. [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose)の投稿、または[コントリビューションガイドライン](https://github.com/hpcaitech/ColossalAI/blob/main/CONTRIBUTING.md)に従ってGitHubでPRを提出
3. 公式提案をメール contact@hpcaitech.com に送信

素晴らしいコントリビューターの皆さまに感謝します！

<a href="https://github.com/hpcaitech/ColossalAI/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hpcaitech/ColossalAI"  width="800px"/>
</a>


<p align="right">(<a href="#top">トップに戻る</a>)</p>


## CI/CD

[GitHub Actions](https://github.com/features/actions)を活用して、開発、リリース、デプロイメントのワークフローを自動化しています。自動化されたワークフローの運用方法については、この[ドキュメント](.github/workflows/README.md)をご確認ください。


## 引用

このプロジェクトはいくつかの関連プロジェクト（私たちのチームおよび他の組織によるもの）にインスパイアされています。[参考文献リスト](./REFERENCE.md)に記載されているこれらの素晴らしいプロジェクトに感謝します。

このプロジェクトを引用するには、以下のBibTeX引用をご使用ください。

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

Colossal-AIはトップカンファレンス [NeurIPS](https://nips.cc/)、[SC](https://sc22.supercomputing.org/)、[AAAI](https://aaai.org/Conferences/AAAI-23/)、
[PPoPP](https://ppopp23.sigplan.org/)、[CVPR](https://cvpr2023.thecvf.com/)、[ISC](https://www.isc-hpc.com/)、[NVIDIA GTC](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-S51482/)等の公式チュートリアルとして採択されています。
