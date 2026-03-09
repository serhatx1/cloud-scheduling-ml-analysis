import json
import os

def generate():
    with open("model_results.json", "r") as f:
        results = json.load(f)
    
    latex_content = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}

\title{Cloud Computing Workload Analysis and Scheduling Prediction}
\author{Fatih Hali\c{c} \& Serhat Arslan}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}
Cloud computing environments require efficient task scheduling to optimize resource utilization and energy consumption. This report presents an analysis of a cloud workload dataset and evaluates the performance of four different machine learning algorithms in predicting the scheduler type used for various tasks.

\section{Dataset Description}
The dataset consists of 5,000 unique task executions with the following key features:
\begin{itemize}
    \item CPU Utilization (\%)
    \item Memory Consumption (MB)
    \item Task Execution Time (ms)
    \item System Throughput (tasks/sec)
    \item Task Waiting Time (ms)
    \item Network Bandwidth Utilization (Mbps)
    \item Job Priority (High, Medium, Low)
\end{itemize}
The target variable is \texttt{Scheduler\_Type}, which includes algorithms like FCFS (First-Come, First-Served), Round Robin, and Priority-Based scheduling.

\section{Methodology}
We implemented and evaluated four classification algorithms:
\begin{enumerate}
    \item \textbf{Random Forest}: An ensemble learning method that constructs multiple decision trees.
    \item \textbf{Support Vector Machine (SVM)}: A model that finds the optimal hyperplane for classification.
    \item \textbf{Logistic Regression}: A statistical model used for multi-class classification using a softmax approach.
    \item \textbf{K-Nearest Neighbors (KNN)}: A non-parametric method that classifies samples based on their proximity to others.
\end{enumerate}

\section{Exploratory Data Analysis}
We performed EDA to understand correlations between resource metrics. 
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{correlation_heatmap.png}
    \caption{Correlation Heatmap of Resource Metrics}
    \label{fig:corr}
\end{figure}

\section{Results}
The performance of the models is summarized in Table \ref{tab:results}.

\begin{table}[h]
    \centering
    \begin{tabular}{lcccc}
        \toprule
        Model & Accuracy & Precision & Recall & F1-Score \\
        \midrule
"""
    for model, metrics in results.items():
        latex_content += f"        {model} & {metrics['Accuracy']:.3f} & {metrics['Precision']:.3f} & {metrics['Recall']:.3f} & {metrics['F1-Score']:.3f} \\\\\n"
    
    latex_content += r"""        \bottomrule
    \end{tabular}
    \caption{Model Performance Comparison}
    \label{tab:results}
\end{table}

\section{Conclusion}
The models demonstrated comparable performance across the balanced dataset. Random Forest emerged as a strong candidate for scheduling prediction tasks in cloud environments.

\end{document}
"""
    
    with open("report.tex", "w") as f:
        f.write(latex_content)
    
    print("LaTeX report generated as 'report.tex'.")

if __name__ == "__main__":
    generate()
