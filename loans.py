import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import List, Dict, Optional
from abc import ABC, abstractmethod


class Loan:
    """Represents a single student loan."""
    def __init__(self, loan_id: str, interest_rate: float,
                 principal_balance: float, current_date: pd.Timestamp):
        self.loan_id = loan_id
        self.interest_rate = interest_rate
        self.principal_balance = principal_balance
        self.interest_accrued = 0.0
        self.last_payment_date = current_date

    @property
    def current_balance(self) -> float:
        """Total balance including principal and accrued interest."""
        return self.principal_balance + self.interest_accrued

    def accrue_interest(self, current_date: pd.Timestamp) -> None:
        """Accrue interest since last payment."""
        days_elapsed = (current_date - self.last_payment_date).days
        daily_interest = self.principal_balance * self.interest_rate / 365
        self.interest_accrued += daily_interest * days_elapsed

    def apply_payment(self, amount: float) -> Dict[str, float]:
        """
        Apply payment to loan (interest first, then principal).
        Returns dict with interest_paid and principal_paid.
        """
        interest_paid = min(self.interest_accrued, amount)
        self.interest_accrued -= interest_paid
        remaining = amount - interest_paid

        principal_paid = min(self.principal_balance, remaining)
        self.principal_balance -= principal_paid

        return {
            'interest_paid': interest_paid,
            'principal_paid': principal_paid,
            'total_paid': interest_paid + principal_paid
        }

    def update_payment_date(self, current_date: pd.Timestamp) -> None:
        """Update the last payment date."""
        self.last_payment_date = current_date

    def is_paid_off(self) -> bool:
        """Check if loan is fully paid off."""
        return self.current_balance <= 0.01  # Small tolerance for floating point


class LoanCollection:
    """Manages a collection of loans."""

    def __init__(self, loans: List[Loan]):
        self.loans = {loan.loan_id: loan for loan in loans}

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, current_date: pd.Timestamp) -> 'LoanCollection':
        """Create portfolio from pandas DataFrame."""
        loans = []
        for _, row in df.iterrows():
            loan = Loan(
                loan_id=row['loan_id'],
                interest_rate=row['interest_rate'],
                principal_balance=row['principal_balance'],
                current_date=current_date
            )
            loan.interest_accrued = row.get('interest_accrued', 0.0)
            loans.append(loan)
        return cls(loans)

    @property
    def total_balance(self) -> float:
        """Total balance across all loans."""
        return sum(loan.current_balance for loan in self.active_loans)

    @property
    def active_loans(self) -> List[Loan]:
        """Get list of loans that aren't paid off."""
        return [loan for loan in self.loans.values() if not loan.is_paid_off()]

    def get_highest_interest_loan(self) -> Optional[Loan]:
        """Get the loan with the highest interest rate."""
        active = self.active_loans
        if not active:
            return None
        return max(active, key=lambda x: x.interest_rate)

    def get_smallest_balance_loan(self):
        active = self.active_loans
        if not active:
            return None
        return min(active, key=lambda x: x.current_balance)

    def accrue_all_interest(self, current_date: pd.Timestamp) -> None:
        """Accrue interest on all active loans."""
        for loan in self.active_loans:
            loan.accrue_interest(current_date)

    def update_all_payment_dates(self, current_date: pd.Timestamp) -> None:
        """Update payment date for all loans."""
        for loan in self.loans.values():
            loan.update_payment_date(current_date)


class PaymentStrategy(ABC):
    """Abstract base class for payment allocation strategies."""

    @abstractmethod
    def allocate_payment(self, portfolio: LoanCollection,
                        payment_amount: float) -> Dict[str, Dict[str, float]]:
        """
        Allocate payment across loans in the portfolio.
        Returns dict mapping loan_id to payment breakdown.
        """
        pass


class AvalanchePaymentStrategy(PaymentStrategy):
    """
    Avalanche method: pay minimums on all loans,
    then extra to highest interest rate loan.
    """

    def __init__(self, minimum_payment: float = 0):
        self.minimum_payment = minimum_payment
        self.name = "Avalanche (Highest Interest First)"

    def allocate_payment(self, portfolio: LoanCollection,
                        payment_amount: float) -> Dict[str, Dict[str, float]]:
        """Allocate payment using avalanche method."""
        results = {}
        remaining = payment_amount

        # Step 1: Apply minimum payment proportionally to all loans
        if self.minimum_payment > 0 and portfolio.total_balance > 0:
            for loan in portfolio.active_loans:
                proportion = loan.current_balance / portfolio.total_balance
                min_payment = proportion * self.minimum_payment
                min_payment = min(min_payment, remaining)

                payment_result = loan.apply_payment(min_payment)
                results[loan.loan_id] = payment_result
                remaining -= payment_result['total_paid']

        # Step 2: Apply extra payment to highest interest loans
        while remaining > 0.01 and portfolio.active_loans:
            highest_loan = portfolio.get_highest_interest_loan()
            if not highest_loan:
                break

            payment_result = highest_loan.apply_payment(remaining)

            # Add to existing results or create new entry
            if highest_loan.loan_id in results:
                results[highest_loan.loan_id]['interest_paid'] += payment_result['interest_paid']
                results[highest_loan.loan_id]['principal_paid'] += payment_result['principal_paid']
                results[highest_loan.loan_id]['total_paid'] += payment_result['total_paid']
            else:
                results[highest_loan.loan_id] = payment_result

            remaining -= payment_result['total_paid']

        return results



class SnowballPaymentStrategy(PaymentStrategy):
    """Snowball method: smallest balance first."""

    def __init__(self, minimum_payment: float = 0):
        self.minimum_payment = minimum_payment
        self.name = "Snowball (Smallest Balance First)"

    def allocate_payment(self, portfolio: LoanCollection, payment_amount: float):
        results = {}
        remaining = payment_amount

        if self.minimum_payment > 0 and portfolio.total_balance > 0:
            for loan in portfolio.active_loans:
                proportion = loan.current_balance / portfolio.total_balance
                min_payment = proportion * self.minimum_payment
                min_payment = min(min_payment, remaining)

                payment_result = loan.apply_payment(min_payment)
                results[loan.loan_id] = payment_result
                remaining -= payment_result['total_paid']

        while remaining > 0.01 and portfolio.active_loans:
            smallest_loan = portfolio.get_smallest_balance_loan()
            if not smallest_loan:
                break

            payment_result = smallest_loan.apply_payment(remaining)

            if smallest_loan.loan_id in results:
                results[smallest_loan.loan_id]['interest_paid'] += payment_result['interest_paid']
                results[smallest_loan.loan_id]['principal_paid'] += payment_result['principal_paid']
                results[smallest_loan.loan_id]['total_paid'] += payment_result['total_paid']
            else:
                results[smallest_loan.loan_id] = payment_result

            remaining -= payment_result['total_paid']

        return results


class LoanSimulator:
    """Simulates loan repayment over time."""

    def __init__(self, portfolio: LoanCollection,
                 payment_strategy: PaymentStrategy,
                 start_date: pd.Timestamp,
                 total_interest_paid_to_date: np.float64,
                 total_principal_paid_to_date: np.float64):
        self.portfolio = portfolio
        self.payment_strategy = payment_strategy
        self.current_date = start_date
        self.records = []
        self.total_interest_paid = total_interest_paid_to_date
        self.total_principal_paid = total_principal_paid_to_date

    def run_simulation(self, payment_schedule: List[float]) -> pd.DataFrame:
        """
        Run the loan repayment simulation.
        Returns DataFrame with simulation results.
        """
        month_idx = 0

        while self.portfolio.total_balance > 0.01 and month_idx < len(payment_schedule):
            # Accrue interest
            self.portfolio.accrue_all_interest(self.current_date)

            # Apply payment
            payment_amount = payment_schedule[month_idx]
            payment_results = self.payment_strategy.allocate_payment(
                self.portfolio, payment_amount
            )

            # Track totals and record results
            self._record_month(payment_results)

            # Update dates and increment
            self.portfolio.update_all_payment_dates(self.current_date)
            month_idx += 1
            self.current_date += pd.DateOffset(months=1)

        return self._create_results_dataframe()

    def _record_month(self, payment_results: Dict[str, Dict[str, float]]) -> None:
        """Record the results for the current month."""
        month_data = []

        for loan_id, loan in self.portfolio.loans.items():
            payment_info = payment_results.get(loan_id, {
                'interest_paid': 0.0,
                'principal_paid': 0.0,
                'total_paid': 0.0
            })

            # Track totals
            self.total_interest_paid += payment_info['interest_paid']
            self.total_principal_paid += payment_info['principal_paid']

            month_data.append({
                'date': self.current_date,
                'loan_id': loan_id,
                'interest_rate': loan.interest_rate,
                'principal_balance': loan.principal_balance,
                'interest_accrued': loan.interest_accrued,
                'current_balance': loan.current_balance,
                'loan_interest_paid': payment_info['interest_paid'],
                'loan_principal_paid': payment_info['principal_paid'],
                'total_payment_amount': payment_info['total_paid']
            })

        self.records.extend(month_data)

    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create final results DataFrame."""
        df = pd.DataFrame(self.records)

        # Round numeric columns
        numeric_cols = ['interest_rate', 'current_balance', 'principal_balance',
                       'interest_accrued', 'loan_interest_paid',
                       'loan_principal_paid', 'total_payment_amount']
        for col in numeric_cols:
            df[col] = df[col].round(2)

        return df

    def print_summary(self) -> None:
        """Print simulation summary."""
        months = len(self.records) // len(self.portfolio.loans)
        final_date = self.current_date - pd.DateOffset(months=1)

        print(f"Payoff completed in {months} months on {final_date.date()}!")
        print(f"Total Principal Paid: ${self.total_principal_paid:,.2f}")
        print(f"Total Interest Paid: ${self.total_interest_paid:,.2f}")
        print(f"Total Paid: ${self.total_principal_paid + self.total_interest_paid:,.2f}")


# ==================== Main Execution ====================

def load_loan_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess loan data."""
    df = pd.read_csv(
        filepath,
        dtype={
            'interest_rate': np.float64,
            'current_balance': np.float64,
            'principal_balance': np.float64,
            'monthly_payment': np.float64
        }
    )

    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[df['date'] == df['date'].max()]  # Take latest data only

    # Calculate interest accrued
    df['interest_accrued'] = df['current_balance'] - df['principal_balance']

    return df


def create_payment_schedule() -> List[float]:
    """Define the payment schedule."""
    schedule = []
    schedule += [1500] * 2          # Jan-Feb 2026
    schedule += [5000]              # March 2026 bonus
    schedule += [2500] * 11         # Apr 2026 - Feb 2027
    schedule += [8000]              # March 2027 bonus
    schedule += [3100] * 11         # Apr 2027 - Feb 2028
    schedule += [5000]              # March 2028 bonus
    schedule += [3500] * 18         # Apr 2028 - Sep 2029
    schedule += [3800] * 100        # Onward
    return schedule

def load_loan_data(filepath: str = "./loan-data-full.csv") -> pd.DataFrame:
    """Load and preprocess loan data."""
    df = pd.read_csv(
        filepath,
        dtype={
            'interest_rate': np.float64,
            'current_balance': np.float64,
            'principal_balance': np.float64,
            'monthly_payment': np.float64
        }
    )

    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[df['date'] == df['date'].max()]
    df['interest_accrued'] = df['current_balance'] - df['principal_balance']

    return df


if __name__ == "__main__":
    # Load data
    df = load_loan_data("./loan-data-full.csv")

    # Create portfolio
    start_date = pd.to_datetime(date(2026, 1, 16))
    portfolio = LoanCollection.from_dataframe(df, start_date)

    # Create payment strategy
    payment_strategy = AvalanchePaymentStrategy(minimum_payment=0)

    # Create and run simulator
    simulator = LoanSimulator(portfolio, payment_strategy, start_date, total_interest_paid_to_date=1246.13, total_principal_paid_to_date=7951.15)
    payment_schedule = create_payment_schedule()

    results = simulator.run_simulation(payment_schedule)
    simulator.print_summary()

    # Save results
    results.to_csv("simulation-results.csv", index=False)
    print(f"\nResults saved to simulation-results.csv")
