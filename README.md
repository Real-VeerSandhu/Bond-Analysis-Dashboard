# Fixed-Income Analysis Dashboard

## Overview

This project is a **Fixed-Income Bond Analysis [Dashboard](https://bond-analysis-dashboard.streamlit.app/)** built using **Python, Streamlit, NumPy, SciPy, and Plotly**. The dashboard provides key insights into bond valuation, risk metrics, and price simulations using Monte Carlo methods.


## Features

- **Yield to Maturity (YTM) Calculation** – Computes the bond's internal rate of return using numerical root-finding methods.
- **Bond Pricing Model** – Calculates bond price based on coupon rate, time to maturity, and YTM.
- **Bond Duration & Convexity** – Measures interest rate sensitivity and curvature risk.
- **Current Yield Calculation** – Determines the annual return based on the bond’s market price.
- **Monte Carlo Simulation** – Generates a distribution of possible bond prices based on simulated YTMs.
- **Interactive Dashboard** – Users can input bond parameters and visualize results dynamically.

## Monte Carlo Simulation Statistics

The Monte Carlo module generates **thousands of bond price simulations**, providing:

- **Mean Price** – Expected bond price.
- **Standard Deviation** – Volatility of simulated prices.
- **5th and 95th Percentiles** – Risk bounds for price movement.
- **Min and Max Prices** – Extreme cases for stress testing.

## Technologies Used

- **Python** – Core programming language.
- **Streamlit** – Interactive web application framework.
- **NumPy & SciPy** – Numerical computing and optimization.
- **Plotly** – Data visualization for histograms and charts.

## Sample Output
![Sample](https://github.com/Real-VeerSandhu/Bond-Analysis-Dashboard/blob/main/demo.png)

## License
This project is licensed under the MIT License – see the LICENSE file for details.
