# AceReason Evaluation Toolkit

We share our evaluation script and code in https://huggingface.co/nvidia/AceReason-Nemotron-14B/blob/main/evaluation.tar.gz

## Environment
- vllm==0.7.3
- torch==2.5.1
- transformers==4.48.2
- 8x NVIDIA H100 80GB HBM3 (CUDA Version: 12.8)

### Dataset Download
LiveCodeBench:
```
from datasets import load_dataset

ds = load_dataset(
    "livecodebench/code_generation_lite",
    version_tag="release_v6",
)["test"]

ds.to_json("data/livecodebench_problems.json", orient="records", lines=False)
```

Math: see data/*

## Evaluation Script

For model generation on single seed, please use the following command:

```
bash generate_livecodebench.sh ${model_path} ${seed} ${output_path} ${model_type}
bash generate_aime.sh ${model_path} ${seed} aime24 ${output_path} ${model_type}
bash generate_aime.sh ${model_path} ${seed} aime25 ${output_path} ${model_type}
```
Please specify model_type as r1 for AceReason-Nemotron-1.0 models, and qwen for AceReason-Nemotron-1.1 models.

Or you can use our configured seeds to reproduce our results on AIME 24/25 (avg@64) and LiveCodeBench v5/v6 (avg@8) as follows:

```
bash run_livecodebench.sh ${model_path} ${output_path}
bash run_aime.sh ${model_path} ${output_path}
```

For benchmark evaluation, we provide the following evaluation command to reproduce our results:

```
python evaluate_livecodebench.py -g ${output_path}
python evaluate_aime.py --modelfolder ${output_path} --test_data data/aime24.jsonl
python evaluate_aime.py --modelfolder ${output_path} --test_data data/aime25.jsonl
```

## Reference Results
We also left our generations into cache.tar.gz as references.

```
LiveCodeBench AceReason-Nemotron-1.0-7B (Avg@8)
=================================================================
Months          Corrects        Total           Accuracy
2023-05         180             272             66.17647058823529
2023-06         238             312             76.28205128205128
2023-07         337             432             78.00925925925925
2023-08         185             288             64.23611111111111
2023-09         275             352             78.125
2023-10         257             352             73.01136363636364
2023-11         217             280             77.5
2023-12         228             320             71.25
2024-01         193             288             67.01388888888889
2024-02         169             256             66.015625
2024-03         234             360             65.0
2024-04         226             296             76.35135135135135
2024-05         211             288             73.26388888888889
05/23-05/24     2950            4096            72.021484375
2024-06         277             368             75.27173913043478
2024-07         223             344             64.82558139534883
2024-08         275             528             52.083333333333336
2024-09         204             376             54.255319148936174
2024-10         209             424             49.29245283018868
2024-11         216             456             47.36842105263158
2024-12         223             392             56.88775510204081
2025-01         161             408             39.46078431372549
06/24-01/25     1788            3296            54.24757281553398
2025-02         179             408             43.872549019607845
2025-03         258             544             47.4264705882353
2025-04         38              96              39.583333333333336
v5              1142            2232            51.16487455197132
v6              621             1400            44.357142857142854

LiveCodeBench AceReason-Nemotron-1.0-14B (Avg@8)
=================================================================
Months          Corrects        Total           Accuracy
2023-05         211             272             77.57352941176471
2023-06         282             312             90.38461538461539
2023-07         393             432             90.97222222222223
2023-08         219             288             76.04166666666667
2023-09         315             352             89.48863636363636
2023-10         294             352             83.52272727272727
2023-11         229             280             81.78571428571429
2023-12         263             320             82.1875
2024-01         219             288             76.04166666666667
2024-02         201             256             78.515625
2024-03         296             360             82.22222222222223
2024-04         252             296             85.13513513513513
2024-05         233             288             80.90277777777777
05/23-05/24     3407            4096            83.1787109375
2024-06         311             368             84.51086956521739
2024-07         248             344             72.09302325581395
2024-08         299             528             56.628787878787875
2024-09         232             376             61.702127659574465
2024-10         266             424             62.735849056603776
2024-11         282             456             61.8421052631579
2024-12         253             392             64.54081632653062
2025-01         217             408             53.18627450980392
06/24-01/25     2108            3296            63.95631067961165
2025-02         211             408             51.71568627450981
2025-03         324             544             59.55882352941177
2025-04         41              96              42.708333333333336
v5              1350            2232            60.483870967741936
v6              775             1400            55.357142857142854

LiveCodeBench AceReason-Nemotron-1.1-7B (Avg@8)
=================================================================
Months          Corrects        Total           Accuracy
2023-05         205             272             75.36764705882354
2023-06         255             312             81.73076923076923
2023-07         356             432             82.4074074074074
2023-08         208             288             72.22222222222223
2023-09         287             352             81.5340909090909
2023-10         278             352             78.97727272727273
2023-11         234             280             83.57142857142857
2023-12         263             320             82.1875
2024-01         215             288             74.65277777777777
2024-02         182             256             71.09375
2024-03         270             360             75.0
2024-04         254             296             85.8108108108108
2024-05         221             288             76.73611111111111
05/23-05/24     3228            4096            78.80859375
2024-06         309             368             83.96739130434783
2024-07         235             344             68.31395348837209
2024-08         292             528             55.303030303030305
2024-09         211             376             56.11702127659574
2024-10         254             424             59.905660377358494
2024-11         269             456             58.99122807017544
2024-12         239             392             60.96938775510204
2025-01         194             408             47.549019607843135
06/24-01/25     2003            3296            60.77063106796116
2025-02         203             408             49.754901960784316
2025-03         306             544             56.25
2025-04         41              96              42.708333333333336
v5              1283            2232            57.482078853046595
v6              726             1400            51.857142857142854

AceReason-Nemotron-7B
====================================
AIME2024 (Avg@64) 68.64583333333334
AIME2025 (Avg@64) 53.59375000000002

AceReason-Nemotron-14B
====================================
AIME2024 (Avg@64) 78.43749999999997
AIME2025 (Avg@64) 67.65625

AceReason-Nemotron-1.1-7B
====================================
AIME2024 (Avg@64) 72.60416666666667
AIME2025 (Avg@64) 64.84375
```
