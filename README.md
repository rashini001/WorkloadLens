# WorkloadLens

**A Machine Learning-Based Academic Workload Pattern Analysis and
RAG-Augmented Time Management Support System for Sri Lankan University
Students**

WorkloadLens is a research prototype designed to help explore academic
workload patterns among Sri Lankan undergraduates and provide
workload-aware, locally relevant time-management guidance. It combines
supervised machine-learning classification with Retrieval-Augmented
Generation (RAG), using a curated knowledge base to ground AI-generated
recommendations.

**Repository:** https://github.com/rashini001/WorkloadLens.git

## Overview

University students may need to manage overlapping assignments,
examinations, projects, lectures, limited study time, employment, sleep
routines, and other responsibilities. Generic time-management advice may
not reflect the different circumstances students face.

WorkloadLens explores an integrated approach:

1.  A student enters a workload profile through the user interface.
2.  A trained machine-learning classifier estimates the workload-risk
    category.
3.  The workload profile and risk category guide retrieval from a
    knowledge base.
4.  A locally hosted large language model generates practical guidance
    from retrieved material and identifies its sources.
5.  A baseline mode generates a response using the same language model
    without retrieved knowledge, allowing comparison.

The project is a research prototype, not an operational university-wide
service.

## Key Features

-   **Workload-risk classification:** Compares Random Forest, XGBoost,
    and Support Vector Machine (SVM) models.
-   **Workload-aware guidance:** Uses the student's submitted profile
    and predicted risk category to guide retrieval.
-   **Three-layer knowledge base:**
    -   Sri Lankan academic and cultural context
    -   Evidence-based workload and time-management interventions
    -   Local institutional resources
-   **Retrieval-Augmented Generation:** Retrieves relevant passages and
    supplies them to the language model as grounding context.
-   **Source-aware responses:** The generation prompt instructs the
    model to use the provided content and name sources.
-   **Baseline comparison:** Compares RAG-supported guidance with a
    response from the same LLM without retrieval.
-   **Research evaluation:** Includes classifier metrics, functional
    testing, response evaluation, performance measurements, and
    usability assessment.

## System Architecture

The research describes four interacting layers:

-   **Application layer:** React-based student form and guidance
    display.
-   **Machine-learning layer:** Data preparation and workload-risk
    prediction.
-   **RAG layer:** Profile-guided retrieval, prompt construction, and
    LLM response generation.
-   **Data and knowledge layer:** Survey data, trained models, and the
    curated knowledge base.

### Workflow

``` text
Student workload profile
          |
          v
Data preparation / feature processing
          |
          v
Workload-risk classifier
(Random Forest / XGBoost / SVM)
          |
          v
Predicted workload-risk category
(Low / Moderate / High)
          |
          v
Profile-guided retrieval from ChromaDB
          |
          v
Prompt construction with retrieved passages
          |
          v
Llama 3 via Ollama
          |
          v
Personalized time-management guidance
with source information
```

A baseline mode uses the same LLM and student profile but does not
include retrieved passages.

## Technology Stack

The thesis identifies the following technologies:

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Python                              Data preparation, machine-learning
                                      models, and pipeline

  Pandas and NumPy                    Data cleaning, encoding, and
                                      analysis

  scikit-learn                        Random Forest, SVM,
                                      cross-validation, and evaluation
                                      metrics

  XGBoost                             Gradient-boosted classifier

  Sentence Transformers               Embedding knowledge-base passages
                                      and queries

  ChromaDB                            Local vector database for
                                      embeddings and metadata

  Ollama with Llama 3                 Local language-model inference

  React                               Student input form and guidance
                                      interface

  GitHub                              Version control for source code,
                                      notebooks, and results
  -----------------------------------------------------------------------

## Research Dataset and Knowledge Base

### Student workload survey

The research reports **957 valid survey responses** from Sri Lankan
undergraduate students at state and non-state institutions. The survey
covers workload-related characteristics such as:

-   Assignment workload and overlapping deadlines
-   Assessment types
-   Available study time
-   Sleep routine
-   Part-time employment
-   Year and field of study
-   Institution type
-   Self-reported time-management outcomes

The thesis describes a held-out test set of 192 responses (approximately
20% of the dataset), with stratified splitting. The data are cleaned and
encoded before model training; numeric features are scaled for SVM where
required.

### RAG knowledge base

The thesis describes **30 documents**, with 10 documents in each of the
three knowledge-base layers listed above. Documents are cleaned and
divided into chunks of approximately 200--400 tokens, embedded using
Sentence Transformers, and stored in ChromaDB with metadata such as
layer, title, source, and year.

> **Data and privacy:** Do not commit survey responses, personally
> identifiable information, API credentials, local model files, or other
> restricted research materials to a public repository. Follow the
> project's ethical-clearance conditions and data-sharing permissions.

## Research Results

The thesis reports the following results:

-   **Best classifier by F1-score:** XGBoost, with an F1-score of
    **0.605** on the held-out test set. The thesis also notes that all
    evaluated classifiers had limited effectiveness in distinguishing
    high-risk from low-risk students.
-   **RAG versus baseline:** 30 raters evaluated responses for
    relevance, actionability, cultural fit, and specificity. The RAG
    system received higher mean ratings across all four criteria, with
    reported differences of 1.05--1.21 points (*p* \< .001).
-   **Usability:** Mean System Usability Scale (SUS) score of **91.08**
    (SD = 10.54; 30 students).
-   **Response time:** The thesis reports an average response time of
    approximately **117.6 seconds** in its test environment.

These are results reported in the dissertation, not a guarantee of
performance on other datasets or hardware. In particular, the
classifier's F1-score and the reported RAG evaluation limitations should
be considered when interpreting the prototype's output.

## Getting Started

The dissertation identifies the main technologies, but this README does
not assume a particular source-code layout or invent project-specific
launch commands. Check the files in the repository for the implemented
entry point, environment configuration, and dependency versions.

### Prerequisites

Based on the documented technology stack, a development environment may
require:

-   Python
-   Node.js and npm for the React interface
-   Ollama
-   The Llama 3 model used by the project
-   Python dependencies for data processing, machine learning,
    embeddings, and ChromaDB

Use the versions and model tag specified by the repository's
configuration files, if provided.

### Setup

1.  Clone the repository:

    ``` bash
    git clone https://github.com/rashini001/WorkloadLens.git
    cd WorkloadLens
    ```

2.  Inspect the repository for its dependency and configuration files
    (for example, `requirements.txt`, `pyproject.toml`, `package.json`,
    environment templates, or setup instructions).

3.  Install dependencies according to the files and instructions
    actually provided in the repository.

4.  If the project uses Ollama, install and start Ollama and obtain the
    Llama model tag expected by the implementation. Confirm that the
    model is available locally before running the application.

5.  Follow the repository's documented commands to prepare the data,
    train or load the classifier, build or load the vector database, and
    start the application.

**Note:** The exact run commands depend on the repository's current
files and configuration. This README intentionally does not specify
unverified script names, dependency versions, or launch commands.

## Evaluation

The dissertation evaluates the prototype through:

-   Functional tests
-   Comparison of Random Forest, XGBoost, and SVM on a held-out test set
-   Classification metrics including precision, recall, F1-score, and
    confusion matrices
-   RAG response evaluation and comparison with a non-retrieval baseline
-   System performance measurements
-   User ratings and the System Usability Scale (SUS)

For reproducibility, use the saved model, data-processing steps, split
strategy, dependency versions, and evaluation scripts provided in the
repository. Do not compare results unless the evaluation conditions are
comparable.

## Scope and Limitations

-   The system is a research prototype and is not integrated with
    university learning-management or information systems.
-   The thesis describes an English-language prototype; Sinhala and
    Tamil support are outside its stated scope.
-   Predictions depend on the quality, representativeness, and
    distribution of the collected survey data.
-   Self-reported survey data may contain recall or response bias.
-   Retrieval-grounded generation can still produce unsupported or
    incomplete information.
-   The reported classifier results indicate that workload-risk
    predictions should be interpreted cautiously.
-   The prototype does not provide clinical diagnosis or treatment, and
    its guidance should not be treated as professional medical advice.

## Responsible Use

WorkloadLens is intended to support research and provide time-management
suggestions, not to make high-stakes decisions about students. Do not
use its risk classifications as definitive assessments of a student's
ability, wellbeing, or academic future. Verify source material and
review generated guidance before relying on it.

## Project Background

This project was developed as part of a BSc (Hons) in Computing research
dissertation. Its research approach follows Design Science Research
Methodology and focuses on designing and evaluating an artefact that
combines workload analysis with contextual academic support.

## Repository

Source code and project materials:

**https://github.com/rashini001/WorkloadLens**

Please refer to the repository for the current implementation,
notebooks, configuration, and any additional usage instructions.
