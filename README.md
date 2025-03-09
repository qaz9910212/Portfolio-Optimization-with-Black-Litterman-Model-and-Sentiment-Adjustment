# Portfolio-Optimization-with-Black-Litterman-Model-and-Sentiment-Adjustment
This repository contains a Python script implementing portfolio optimization using the Black-Litterman model with an extended Bayesian approach. The model adjusts asset allocation weights based on investor sentiment to enhance portfolio performance.
Features

Black-Litterman Model Extension: Adjusts investor views based on sentiment data.

Markowitz Mean-Variance Optimization: Computes optimal asset weights.

CAPM-Based Factor Model: Estimates asset returns using Ordinary Least Squares (OLS) regression.

Dynamic Tau Adjustment: Adjusts the uncertainty parameter based on market sentiment.

Rolling Window Analysis: Applies a moving window approach for time-series optimization.

Files

30indffn.py: Main script that loads financial data, performs regression, and computes portfolio weights.

Data_BL.xlsx: Input dataset (not included in the repository) containing asset returns, market returns, and sentiment scores.

Logistic_Results.xlsx: Output file containing regression results for logistic models.

Installation

Ensure you have Python installed along with the required dependencies. You can install the necessary packages using:

pip install numpy pandas statsmodels openpyxl

Usage

Modify the file_path variable in 30indffn.py to point to your dataset and execute the script:

python 30indffn.py

The script will process the historical financial data, estimate asset returns, and generate optimal portfolio allocations.

Output

A detailed regression results report.

Optimal portfolio weights based on sentiment-adjusted Black-Litterman estimates.

Processed data saved to Logistic_Results.xlsx for further analysis.

Author
Chen Weilin
National Central University
