import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from textwrap import dedent
from typing import List
import sys
import os
import plotly.express as px

# Import loan classes (assumes loans.py is in same directory)
try:
    from loans import (
        Loan, LoanCollection, AvalanchePaymentStrategy,
        SnowballPaymentStrategy, LoanSimulator
    )
except ImportError:
    st.error("Error: Could not import from loans.py. Make sure loans.py is in the same directory.")
    st.stop()


def create_payment_schedule(num_years=10, default_payment=1600.00, advanced=False, custom_schedule=None) -> List[float]:
    """Define the payment schedule."""
    if advanced and custom_schedule is not None:
        return custom_schedule

    # Create constant payment schedule
    num_months = num_years * 12
    schedule = [default_payment] * num_months
    return schedule


def load_loan_data(df: pd.DataFrame, current_date: pd.Timestamp) -> LoanCollection:
    """Create LoanCollection from DataFrame."""
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
    return LoanCollection(loans)


# Initialize session state for payment schedule
if "df" not in st.session_state:
    start_date = pd.to_datetime(date(2026, 1, 16))
    payments = create_payment_schedule(10, 1600.00)
    dates = [start_date + relativedelta(months=i) for i in range(len(payments))]

    st.session_state.df = pd.DataFrame({
        "Payment Date": dates,
        "Amount": payments
    })

# Initialize Tab 2 working copy (stores user's UI edits)
if "tab2_working_df" not in st.session_state:
    st.session_state.tab2_working_df = st.session_state.df.copy()

# Initialize Tab 3 custom schedule (stores code-based schedules)
if "tab3_custom_df" not in st.session_state:
    st.session_state.tab3_custom_df = st.session_state.df.copy()


st.set_page_config(page_title="Student Loan Repayment Planner", layout="wide")
st.title("Student Loan Repayment Planner")

# Create tabs
tab1, tab2, tab3, tab0 = st.tabs(["Loan Simulation", "Edit Payment Schedule", "Edit Payment Schedule (Advanced)", "README"])


# ==================== TAB 0: README ====================
with tab0:

    st.markdown("""
    This tool helps you simulate different loan repayment strategies and understand the total cost of paying off your student loans under various scenarios.
    """)

    st.subheader("📋 Getting Started")

    st.markdown("""
    **Step 1: Gather Your Loan Information**
    - Log in to your loan servicer's payment portal
    - For each individual loan, record:
      - **Loan ID**: A unique identifier for the loan (e.g., "Loan A", "Federal Stafford 1")
      - **Principal Balance**: The original loan amount (what you originally borrowed)
      - **Current Balance**: The amount you still owe today
      - **Interest Rate**: The annual interest rate (as a decimal, e.g., 0.0754 for 7.54%)
      - **Current Date**: The date when these balances were recorded

    **Step 2: Prepare Your CSV File**
    Create a CSV file with your loan data. Here's an example:
    """)

    # Create and display example dataframe
    example_data = {
        'loan_id': ['Loan A', 'Loan B', 'Loan C'],
        'principal_balance': [15000.00, 22500.00, 8750.00],
        'current_balance': [14250.00, 21450.00, 8500.00],
        'interest_rate': [0.0754, 0.0620, 0.0485],
        'date': ['2026-01-16', '2026-01-16', '2026-01-16']
    }
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df, use_container_width=True)

    st.markdown("""
    **Required Columns:**
    - `loan_id` (string): Unique identifier for each loan
    - `principal_balance` (float): Total remaining principal owed on this loan
    - `current_balance` (float): Total amount currently owed on this loan (i.e. sum of principal + interest)
    - `interest_rate` (float): Annual interest rate as a decimal (e.g., 0.0754)
    - `date` (string): Date in YYYY-MM-DD format when balances were recorded
    """)

    st.subheader("🎯 How the App Works")

    st.markdown("""
    **1. Loan Simulation Tab**
    - **Configure Parameters**: Select your simulation start date and repayment strategy
      - *Avalanche*: Pay off highest interest loans first (saves the most interest)
      - *Snowball*: Pay off smallest balance loans first (psychological wins)
    - **Set Payment Schedule**: Choose between:
      - *Standard Repayment*: Fixed monthly payment (default $1,600.00)
      - *Custom Repayment Plan*: Use a customized schedule created within the "Edit Payment Schedule" tab
      - *Advanced Custom Repayment Plan*: Use a customized schedule created within the Edit Payment Schedule (Advanced) tab
    - **Payment History**: Enter any principal and interest you've already paid for accurate final calculations
    - **Upload Data**: Select your CSV file or use the default to play around (my grad school loan data! 🤮)
    - **Run Simulation**: Get detailed results showing when you'll be debt-free and total amounts paid

    **2. Edit Payment Schedule Tab**
    - Modify your payment schedule using an interactive editor
    - Adjust start date and regenerate the default constant payment schedule
    - Edit individual payment amounts directly

    **3. Advanced Payment Schedule Editor Tab**
    - Write Python code to define complex payment schedules
    - Helpful for creating year-specific payment plans or accounting for one-off larger payments
    - Use either direct list format:
        ```
        schedule = [1500, 1500, 5000, ...]
        ```
    - Or build incrementally:
        ```
        schedule = []
        schedule += [1500] * 2
        schedule += [5000] * 10
        ...
        ```
    """)

    st.subheader("📊 Understanding the Results")

    st.markdown("""
    After running a simulation, you'll see:
    - **Total Principal Paid**: Amount paid toward the original loan balance
    - **Total Interest Paid**: Amount paid in interest charges
    - **Total Paid**: Grand total of principal + interest
    - **Months to Payoff**: How long until all loans are paid off
    - **Loan Balance Chart**: Visual representation of how each loan's balance decreases over time
    - **Detailed Results Table**: Month-by-month breakdown with interest rates and balances
    """)

    st.subheader("💡 Tips for Best Results")

    st.markdown("""
    - **Be accurate with current balances**: The more precise your starting data, the better your simulation
    - **Try different strategies**: Compare Avalanche vs Snowball to see which saves more money
    - **Test different payment amounts and custom schedules**: Experiment with a standard repayment plan versus custom payment schedules
    - **Consider lump sums**: Create a custom schedule that includes extra payments when you expect bonuses or tax refunds
    - **Track progress**: Re-run simulations periodically with updated loan balances from your servicer as you make payments
    """)

    st.subheader("❓ FAQ")

    st.markdown("""
    **Q: What if I don't have a CSV file?**
    A: The app can load a default file (loan-data-full.csv) if available. Otherwise, create a simple CSV in Excel or Google Sheets with the required columns.

    **Q: Can I change my payment plan mid-simulation?**
    A: Create a custom payment schedule in the "Edit Payment Schedule" tab that reflects your desired schedule and ensure that the Custom Payment Plan is selected in the Loan Simulator tab.

    **Q: Why are interest accruals not matching my servicer?**
    A: This simulator uses simplified daily interest calculations. Your servicer may use different compounding methods or have specific rules.

    **Q: How do I export my results?**
    A: After running a simulation, use the "Download Results as CSV" button to save the detailed month-by-month breakdown.
    """)


# ==================== TAB 1: Loan Simulation ====================
with tab1:
    st.header("Loan Repayment Simulator")

    # Help info box
    st.info("❓ **New here?** Check out the **README** tab above for getting started, instructions, and FAQs!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Configure simulation parameters")

        # Date picker for start date
        start_date = st.date_input(
            "Simulation Start Date",
            value=date(2026, 1, 16),
            label_visibility="visible",
            key="sim_start_date"
        )
        start_date = pd.to_datetime(start_date)

        # Radio button for payment strategy
        strategy_choice = st.radio(
            "Payment Strategy",
            options=["Avalanche (Highest Interest First)", "Snowball (Smallest Balance First)"],
            index=0
        )
        st.subheader("2. Set Default Payment Amount")

        # Schedule type selection
        schedule_type = st.radio(
            "Schedule Type",
            options=["Standard Repayment", "Custom Repayment Plan", "Advanced Mode Payment Plan"],
            index=0,
            label_visibility="visible"
        )

        if schedule_type == "Standard Repayment":
            constant_payment = st.number_input(
                "Monthly Payment Amount",
                value=st.session_state.get("sim_constant_payment", 1600.00),
                min_value=0.0,
                step=10.0,
                label_visibility="visible",
                key="sim_constant_payment"
            )

        st.subheader("3. Upload loan data file")
        # File uploader for loan data
        uploaded_file = st.file_uploader(
            "Upload Loan Data (CSV)",
            type=["csv"],
            help="CSV should contain: loan_id, interest_rate, principal_balance, current_balance"
        )


    with col2:
        st.subheader("4. Add Aggregate Payment History")

        # Prior payment inputs
        prior_principal = st.number_input(
            "Total Principal Paid to Date",
            value=7951.15,
            min_value=0.0,
            step=100.0,
            label_visibility="visible"
        )
        prior_interest = st.number_input(
            "Total Interest Paid to Date",
            value=1246.13,
            min_value=0.0,
            step=100.0,
            label_visibility="visible"
        )

        st.divider()


    st.divider()

    # Run simulation button
    if st.button("Run Simulation", type="primary", use_container_width=True):
        # Use uploaded file or try to load default
        if uploaded_file is not None:
            loan_df = pd.read_csv(uploaded_file)
        else:
            # Try to load default loan data file
            default_file_path = "loan-data-full.csv"
            try:
                loan_df = pd.read_csv(default_file_path)
                st.info(f"Using default loan data from '{default_file_path}'")
            except FileNotFoundError:
                st.error(f"No file uploaded and default file '{default_file_path}' not found. Please upload a CSV file with loan data.")
                st.stop()

        try:
            # Validate required columns
            required_cols = ['loan_id', 'interest_rate', 'principal_balance']
            missing_cols = [col for col in required_cols if col not in loan_df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Calculate interest_accrued if not present
                if 'interest_accrued' not in loan_df.columns:
                    if 'current_balance' in loan_df.columns:
                        loan_df['interest_accrued'] = loan_df['current_balance'] - loan_df['principal_balance']
                    else:
                        loan_df['interest_accrued'] = 0.0

                # Create portfolio
                portfolio = load_loan_data(loan_df, start_date)

                # Select strategy
                if "Avalanche" in strategy_choice:
                    strategy = AvalanchePaymentStrategy(minimum_payment=0)
                else:
                    strategy = SnowballPaymentStrategy(minimum_payment=0)

                # Get payment schedule from configuration
                if schedule_type == "Standard Repayment":
                    # Generate constant payment schedule
                    num_months = 10 * 12 + 2  # 30 years default
                    payment_schedule = [constant_payment] * num_months
                elif schedule_type == "Advanced Mode Payment Plan":
                    # Use the advanced custom schedule
                    payment_schedule = st.session_state.tab3_custom_df['Amount'].tolist()
                else:
                    # Use the custom repayment plan from Tab 2
                    payment_schedule = st.session_state.tab2_working_df['Amount'].tolist()

                # Run simulator
                simulator = LoanSimulator(
                    portfolio=portfolio,
                    payment_strategy=strategy,
                    start_date=start_date,
                    total_interest_paid_to_date=prior_interest,
                    total_principal_paid_to_date=prior_principal
                )

                results_df = simulator.run_simulation(payment_schedule)

                # Display results
                st.success("Simulation completed!")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Principal Paid", f"${simulator.total_principal_paid:,.2f}")
                with col2:
                    st.metric("Total Interest Paid", f"${simulator.total_interest_paid:,.2f}")
                with col3:
                    total_paid = simulator.total_principal_paid + simulator.total_interest_paid
                    st.metric("Total Paid", f"${total_paid:,.2f}")
                with col4:
                    months = len(results_df) // len(portfolio.loans)
                    st.metric("Months to Payoff", months)

                # Balance over time chart
                st.subheader("Loan Balance Over Time")
                fig = px.line(
                    results_df,
                    x="date",
                    y="current_balance",
                    color="loan_id",
                    title="Remaining Balance Over Time by Loan"
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Current Balance",
                    legend_title="Loan ID",
                    hovermode="x unified",
                    yaxis=dict(tickformat="$,.0f")
                )
                fig.update_traces(hovertemplate="<b>Loan: %{fullData.name}</b><br>Balance: $%{y:,.0f}<extra></extra>")
                st.plotly_chart(fig, use_container_width=True)

                # Detailed results
                st.subheader("Detailed Results")
                results_display = results_df.copy()
                results_display['date'] = results_display['date'].dt.strftime('%Y-%m-%d')
                results_display['interest_rate'] = results_display['interest_rate'].apply(lambda x: f"{x:.4f}")
                st.dataframe(results_display, use_container_width=True)

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="simulation_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error running simulation: {str(e)}")


# ==================== TAB 2: Payment Schedule ====================
with tab2:
    st.header("Payment Schedule Editor")

    # Configuration section
    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    with col1:
        schedule_start_date = st.date_input(
            "Payment Schedule Start Date",
            value=date(2026, 1, 16),
            label_visibility="visible",
            key="schedule_start_date"
        )

    with col2:
        payment_amount = st.number_input(
            "Default Monthly Payment Amount",
            value=st.session_state.get("sim_constant_payment", 1600.00),
            min_value=0.0,
            step=10.0,
            label_visibility="visible",
            key="edit_payment_amount"
        )

    with col3:
        regenerate_schedule = st.button("Regenerate Default", type="secondary", use_container_width=True)

    # Regenerate default schedule
    if regenerate_schedule:
        schedule_start_date_ts = pd.to_datetime(schedule_start_date)
        payments = create_payment_schedule(10, st.session_state.edit_payment_amount)
        new_dates = [schedule_start_date_ts + relativedelta(months=i) for i in range(len(payments))]
        new_df = pd.DataFrame({
            "Payment Date": new_dates,
            "Amount": payments
        })
        st.session_state.tab2_working_df = new_df
        st.session_state.df = new_df.copy()
        st.success("Default payment schedule regenerated!")
        st.rerun()

    st.divider()

    # Display and edit the dataframe (shows working copy, not the current active schedule)
    edited_df = st.data_editor(
        st.session_state.tab2_working_df,
        use_container_width=True,
        num_rows="dynamic"
    )

    # Save custom plan button
    if st.button("Save Custom Payment Plan", type="primary", use_container_width=True, key="tab2_save"):
        st.session_state.tab2_working_df = edited_df
        st.session_state.df = edited_df.copy()
        st.success("Custom payment plan saved!")
        st.rerun()

    st.divider()

    # Display summary statistics (shows current active schedule)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Payments", len(st.session_state.df))
    with col2:
        st.metric("Total Amount", f"${st.session_state.df['Amount'].sum():,.2f}")
    with col3:
        st.metric("Average Payment", f"${st.session_state.df['Amount'].mean():,.2f}")

    # Display the data as a table (shows current active schedule)
    st.subheader("Payment Schedule Preview")
    schedule_display = st.session_state.df.copy()
    schedule_display['Payment Date'] = schedule_display['Payment Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(schedule_display, use_container_width=True)


# ==================== TAB 3: Edit Payment Schedule (Advanced) ====================
with tab3:
    st.header("Edit Payment Schedule (Advanced)")

    st.markdown("Define a custom payment schedule by writing a Python list of payment amounts.")

    # Start date picker
    adv_schedule_start_date = st.date_input(
        "Payment Schedule Start Date",
        value=date(2026, 1, 16),
        label_visibility="visible",
        key="adv_schedule_start_date"
    )

    st.divider()

    # Show examples
    st.markdown("**Example 1:**")
    st.code("schedule = [1500, 1500, 5000, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 8000, 3100]", language="python")

    st.markdown("**Example 2:**")
    st.code(dedent(
        """
        schedule = []
        schedule += [1500] * 2       # months 1-2
        schedule += [5000]           # month 3
        schedule += [2500] * 11      # etc.
        schedule += [8000]
        schedule += [3100] * 11
        schedule += [3500] * 24
        """), language="python")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.write("")  # Spacer

    with col2:
        st.write("")  # Spacer

    # Code input area
    code_input = st.text_area(
        "Custom Payment Schedule (Python list)",
        value=dedent("""
            schedule = []
            schedule += [1500] * 2       # months 1-2
            schedule += [5000]           # month 3
            schedule += [2500] * 11      # etc.
            schedule += [8000]
            schedule += [3100] * 11
            schedule += [3500] * 24
            """).strip(),
        height=200,
        label_visibility="visible",
        placeholder="[1500, 1500, 5000, ...] or schedule = [...]; schedule += [...]"
    )
    st.write("**⚠️ Tip:** Ensure the \"Advanced Mode Payment Plan\" button is selected in Tab 1 if you wish to use this custom plan!")

    # Apply advanced changes
    if st.button("Save Custom Payment Plan", type="primary", use_container_width=True, key="tab3_save"):
        try:
            custom_schedule = None

            # First, try to execute as a script (handles "schedule = ... ; schedule += ..." format)
            try:
                local_namespace = {}
                exec(code_input, {}, local_namespace)
                if 'schedule' in local_namespace:
                    custom_schedule = local_namespace['schedule']
            except:
                pass

            # If that didn't work, try to eval as a direct list
            if custom_schedule is None:
                custom_schedule = eval(code_input)

            # Validate it's a list
            if not isinstance(custom_schedule, list):
                st.error("❌ Input must be a Python list or schedule variable (e.g., [1500, 5000, ...] or schedule = [...]; schedule += [...])")
            else:
                # Validate all elements are numbers
                if not all(isinstance(x, (int, float)) for x in custom_schedule):
                    st.error("❌ All list elements must be numbers")
                else:
                    # Apply the custom schedule (save to Tab 3 specific storage)
                    schedule_start_date_ts = pd.to_datetime(adv_schedule_start_date)
                    new_dates = [schedule_start_date_ts + relativedelta(months=i) for i in range(len(custom_schedule))]
                    new_schedule_df = pd.DataFrame({
                        "Payment Date": new_dates,
                        "Amount": custom_schedule
                    })
                    st.session_state.tab3_custom_df = new_schedule_df
                    st.session_state.df = new_schedule_df.copy()
                    st.success(f"✓ Custom payment schedule saved! ({len(custom_schedule)} payments)")
                    st.rerun()
        except SyntaxError as e:
            st.error(f"❌ Syntax error in your code: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error parsing code input: {str(e)}")

    st.divider()

    # Display current schedule
    st.subheader("Current Payment Schedule")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Payments", len(st.session_state.df))
    with col2:
        st.metric("Total Amount", f"${st.session_state.df['Amount'].sum():,.2f}")
    with col3:
        st.metric("Average Payment", f"${st.session_state.df['Amount'].mean():,.2f}")

    adv_schedule_display = st.session_state.df.copy()
    adv_schedule_display['Payment Date'] = adv_schedule_display['Payment Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(adv_schedule_display, use_container_width=True)
