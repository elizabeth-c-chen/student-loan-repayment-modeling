# Student Loan Repayment Simulator

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

#### 1. Clone or download the project
Ensure you have `app.py`, `loans.py`, and `requirements.txt` in the same directory.

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### 3. Prepare your loan data (optional)
Create a CSV file with your loan information or use the default `loan-data-full.csv` to preview functionality:

**Required columns:**
- `loan_id` (string): Unique identifier for each loan
- `principal_balance` (float): Total remaining principal owed
- `current_balance` (float): Total amount owed (principal + interest)
- `interest_rate` (float): Annual interest rate as a decimal (e.g., 0.0754)
- `date` (string): Date in YYYY-MM-DD format

**Example CSV:**
```
loan_id,principal_balance,current_balance,interest_rate,date
Loan A,15000.00,14250.00,0.0754,2026-01-16
Loan B,22500.00,21450.00,0.0620,2026-01-16
Loan C,8750.00,8500.00,0.0485,2026-01-16
```

### Running the App

```bash
streamlit run app.py
```

The app will launch in your default web browser at `http://localhost:8501`
