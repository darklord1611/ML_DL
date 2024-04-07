### Overview

- data drift -> data not in standard conditions
- proof of concept -> production gap (POC)
- ML production cycle:
  - scoping -> define projects
  - data -> define data, establish baseline, label and organize data
  - modeling -> select and train data, perform error analysis
  - deployment -> deploy in production, monitor & maintain system
- consist of many iterations between step 2-3-4

- scoping:
  - decide what problem to work on
  - key metrics: accuracy, latency, throughput
  - estimated resources and timeline
- data:
  - define data
  - data cleaning
  - data normalization
  - data labeling
- modeling:
  - code(algorithm/model)
  - hyperparameters
  - data
  - product team -> optimize hyperparameters and data
- deployment:

  - edge devices: laptop, PC
  - send requests -> prediction models on the cloud -> return responses

- MLOps: an emerging discipline, compromises a set of tools, methods, principles to support progress of a ML project lifecycle
- key challenges with deployment:
    - concept drift -> the nature of the problem change, data drift -> change in data distribution
    - SE issues: real-time or batch, clouds or edge/browser, compute resources(CPU/GPU/memory), latency & throughput(QPS), logging, security & privacy
- common deployment cases: gradually ramp up with monitoring and rollback
    - new product or capability
    - automate/assist with manual task
    - replace previous ML system
- shadow mode deployment: compare human's performance with model's performance until there is a negligible difference 
- canary deployment: use model for a small fraction of traffic -> gradually increase based on result metrics -> spot early mistakes
- blue green deployment: use an older and new version of the model, flexibly switching a fraction between them -> enable easy rollback 
- degree of automation: human only -> shadow mode -> AI assistance -> partial automation -> full automation

### Monitoring
- use a dashboard, define few statistics/metrics to evaluate, start with many -> refine to a small number of metrics
- some type of metrics:
    - software metrics: memory, throughtput, latency, serverload
    - input metrics
    - output metrics

### Modeling
- algorithms/model + hyperparameters + data -> ML model 
- challenges?
    - doing well on training sets
    - doing well on dev/test sets
    - doing well according to business/project metrics
