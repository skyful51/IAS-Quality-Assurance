#!/usr/bin/env python3
"""
Statistical Correlation Analysis Script (Pearson r, p-value, Spearman rho)
Comparing Image Quality Metrics (FID, KID, LPIPS, SSIM, PSNR, Ours Q) vs
Downstream Task Metrics (Image AUROC, Pixel AUROC, Image AP, Pixel AP, PRO)
across 15 MVTec AD Categories and 3 Synthetic Algorithms (AnomalyDiffusion, RealNet, CutPaste).

Usage:
    python compute_correlations.py \
        --quality_csv /home/jhkang51/Public/IAS-Quality-Assurance/ias_quality_ranking.csv \
        --downstream_csv /home/jhkang51/Public/IAS-Quality-Assurance/asbench_downstream_task.csv \
        --output_csv /home/jhkang51/Public/IAS-Quality-Assurance/correlation_analysis_results.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats


def load_and_preprocess_data(quality_csv, downstream_csv):
    df_quality = pd.read_csv(quality_csv)
    df_downstream = pd.read_csv(downstream_csv)

    # Filter out summary/average rows
    df_quality = df_quality[~df_quality["category"].astype(str).str.lower().isin(["average", "macro average"])].copy()
    df_downstream = df_downstream[~df_downstream["category"].astype(str).str.lower().isin(["average", "macro average"])].copy()

    quality_metrics = ["FID", "KID", "LPIPS", "SSIM", "PSNR", "Ours"]
    downstream_metrics = ["Image AUROC", "Pixel AUROC", "Image AP", "Pixel AP", "PRO"]
    algorithms = ["anomaly-diffuison", "realnet", "cutpaste"]

    # Reshape Quality Data into Long Format: (category, algorithm, metric_name, value)
    records_quality = []
    for idx, row in df_quality.iterrows():
        cat = row["category"].strip()
        for col in df_quality.columns:
            if col == "category":
                continue
            if "/" in col:
                metric_name, algo_name = col.split("/", 1)
                records_quality.append({
                    "category": cat,
                    "algorithm": algo_name.strip(),
                    "metric": metric_name.strip(),
                    "quality_value": float(row[col])
                })
    df_q_long = pd.DataFrame(records_quality)

    # Reshape Downstream Data into Long Format
    records_downstream = []
    for idx, row in df_downstream.iterrows():
        cat = row["category"].strip()
        for col in df_downstream.columns:
            if col == "category":
                continue
            if "/" in col:
                metric_name, algo_name = col.split("/", 1)
                records_downstream.append({
                    "category": cat,
                    "algorithm": algo_name.strip(),
                    "downstream_metric": metric_name.strip(),
                    "downstream_value": float(row[col])
                })
    df_d_long = pd.DataFrame(records_downstream)

    # Merge on category and algorithm
    df_merged = pd.merge(df_q_long, df_d_long, on=["category", "algorithm"])
    return df_merged, quality_metrics, downstream_metrics


def compute_correlations(df_merged, quality_metrics, downstream_metrics):
    results = []

    for q_m in quality_metrics:
        for d_m in downstream_metrics:
            sub = df_merged[(df_merged["metric"] == q_m) & (df_merged["downstream_metric"] == d_m)].dropna()
            
            x = sub["quality_value"].values
            y = sub["downstream_value"].values

            if len(x) > 2:
                p_r, p_p = stats.pearsonr(x, y)
                s_r, s_p = stats.spearmanr(x, y)
            else:
                p_r, p_p, s_r, s_p = np.nan, np.nan, np.nan, np.nan

            results.append({
                "Quality_Metric": q_m,
                "Downstream_Metric": d_m,
                "N": len(sub),
                "Pearson_r": p_r,
                "Pearson_p_value": p_p,
                "Spearman_rho": s_r,
                "Spearman_p_value": s_p
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Compute Pearson & Spearman correlations between Quality Metrics and Downstream Tasks")
    parser.add_argument("--quality_csv", type=str, default="/home/jhkang51/Public/IAS-Quality-Assurance/ias_quality_ranking.csv")
    parser.add_argument("--downstream_csv", type=str, default="/home/jhkang51/Public/IAS-Quality-Assurance/asbench_downstream_task.csv")
    parser.add_argument("--output_csv", type=str, default="/home/jhkang51/Public/IAS-Quality-Assurance/correlation_analysis_results.csv")
    args = parser.parse_args()

    print("==========================================================================")
    print(" Statistical Correlation Analysis: Quality Metrics vs Downstream Performance")
    print(f" Quality CSV   : {args.quality_csv}")
    print(f" Downstream CSV: {args.downstream_csv}")
    print("==========================================================================")

    df_merged, q_metrics, d_metrics = load_and_preprocess_data(args.quality_csv, args.downstream_csv)
    df_corr = compute_correlations(df_merged, q_metrics, d_metrics)

    # Save to CSV
    df_corr.to_csv(args.output_csv, index=False)
    print(f"\nSaved correlation results to: {args.output_csv}\n")

    # Print Summary Table
    print(df_corr.to_string(index=False))


if __name__ == "__main__":
    main()
