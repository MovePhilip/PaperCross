# PaperCross

This repository contains a released benchmark datasets for RAG in scientific document processing field

![](main.png)

## Dataset  statistics

| Item             | Count | Task               | Count |
|------------------|-------|--------------------|-------|
| Task question    | 53    | Table2chart        | 404   |
| Dataset question | 184   | Table2table        | 788   |
| Leaderboard with 5 paper  | 53    | Table2text         | 692   |
| Leaderboard with 6 paper  | 68    | Text2chart         | 362   |
| Leaderboard with 7 paper  | 63    | Text2table         | 732   |
| Unique paper     | 1002  | Text2text          | 666   |
| Total page       | 14064 | Total 2-Hop question | 3644 |

## Dataset files

task.json for task centric question

rank.json for dataset centric question

2hop.json for 2-hop question

## Exampel question
Dataset Centric question
```
Please help me find the performance of methods proposed in different papers on the Accuracy (%) metric for the Unsupervised Image Classification task on the ImageNet (ImageNet) dataset, list the top three metric result. 

Requirements:
1.for each paper, you should only give one metric result (the highest one) of its own proposed method, since most papers will compare the performance of methods proposed in other papers and some variants of the same method in abalation study. 
2. The top three metric result means that you need to find at least three papers that have reported the metric result.
3. The article ID must correspond to the method name, meaning the article ID should refer to the paper in which the method was originally proposed.
4. If a method yields different results under the same experimental conditions in different papers, you should take the results reported in the original paper as the standard.

You need to give the result in JSON format:
[
    {"rank_id":1
    "method":"model name",
    "value": "metric value"
    "paper_id":"paper id"
    },
    {"rank_id":2
    "method":"model name",
    "value": "metric value"
    "paper_id":"paper id"
    },
    {"rank_id":3
    "method":"model name",
    "value": "metric value"
    "paper_id":"paper id"
    }
]                    "

```


## Example inference log of ReAct method

task_example.json for task centric question

rank_example.json for dataset centric question

2hop_example.json for 2-hop question
