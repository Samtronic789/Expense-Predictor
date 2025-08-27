import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

ESSENTIAL_CATEGORIES = set([
    "Groceries", "Utilities", "Rent", "Healthcare",
    "Transport (Work)", "Debt Repayment", "Insurance", "Savings"
])

KEYWORDS = {
    "Groceries": ["keells", "cargills", "arpico"],
    "Utilities": ["ceb", "leco", "water board", "slt", "dialog", "mobitel"],
    "Rent": ["landlord"],
    "Healthcare": ["healthguard", "osu sala", "pharmacy"],
    "Transport (Work)": ["pickme", "uber"],
    "Debt Repayment": ["credit card payment"],
    "Insurance": ["insurance"],
    "Savings": ["fixed deposit"],
    "Dining Out": ["kfc", "mcdonald", "pizza hut", "burger king", "restaurant"],
    "Takeaway": ["uber eats", "pickme food", "food"],
    "Shopping": ["daraz", "aliexpress", "arpico home", "shop"],
    "Entertainment": ["netflix", "spotify", "steam", "cinema"],
    "Subscriptions (Nonessential)": ["youtube premium", "apple tv"],
    "Travel (Leisure)": ["booking.com", "agoda", "airbnb"],
    "Hobbies": ["hobby"],
    "Misc Nonessential": ["coffee", "dessert"]
}

def categorize(desc: str) -> str:
    if not isinstance(desc, str) or pd.isna(desc):
        return "Other"
    d = str(desc).lower()
    for cat, kws in KEYWORDS.items():
        for kw in kws:
            if kw in d:
                return cat
    return "Other"

def is_nonessential(category: str) -> bool:
    return category not in ESSENTIAL_CATEGORIES and category != "Other"

def forecast_next_month(monthly_series: pd.Series) -> float:
    """Improved forecasting with better error handling"""
    if len(monthly_series) == 0:
        return 0.0
    
    y = monthly_series.values.astype(float)
    y = y[~np.isnan(y)]  # Remove NaN values
    
    if len(y) >= 3:
        x = np.arange(len(y))
        try:
            coeffs = np.polyfit(x, y, 1)
            pred = coeffs[0] * len(y) + coeffs[1]
        except np.RankWarning:
            pred = float(y.mean())
    elif len(y) >= 1:
        pred = float(y.mean())
    else:
        pred = 0.0
    return max(0.0, float(pred))

def month_start(dt: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=dt.year, month=dt.month, day=1)

def month_progress(current_date: pd.Timestamp) -> float:
    import calendar
    total_days = calendar.monthrange(current_date.year, current_date.month)[1]
    return min(current_date.day / total_days, 1.0)

def process_csv(input_csv: str, out_dir: str = "./outputs"):
    """Enhanced CSV processing with better error handling - categories plot removed"""
    try:
        os.makedirs(out_dir, exist_ok=True)
        plot_forecast_path = os.path.join(out_dir, "nonessential_forecast_plot.png")
        summary_csv_path = os.path.join(out_dir, "monthly_summary.csv")
        alerts_path = os.path.join(out_dir, "alerts.txt")
        forecast_json_path = os.path.join(out_dir, "forecast.json")

        # Read CSV with better error handling
        try:
            tx = pd.read_csv(input_csv)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
        
        # Validate required columns
        required_cols = ["date", "description", "type", "amount"]
        missing_cols = [col for col in required_cols if col not in tx.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Parse dates with error handling
        try:
            tx["date"] = pd.to_datetime(tx["date"], errors='coerce')
            if tx["date"].isna().any():
                raise ValueError("Some dates could not be parsed")
        except Exception as e:
            raise ValueError(f"Error parsing dates: {str(e)}")
        
        # Clean and process data
        tx = tx.dropna(subset=["date", "amount"])
        tx["amount"] = pd.to_numeric(tx["amount"], errors='coerce')
        tx = tx.dropna(subset=["amount"])
        
        if len(tx) == 0:
            raise ValueError("No valid transactions found after cleaning data")
        
        tx["category"] = tx["description"].apply(categorize)
        tx["is_nonessential"] = tx["category"].apply(is_nonessential)

        # Filter expenses (handle case variations)
        expenses = tx[tx["type"].str.lower().isin(["debit", "expense"])].copy()
        if len(expenses) == 0:
            raise ValueError("No expense transactions found")
        
        expenses["year_month"] = expenses["date"].dt.to_period("M").dt.to_timestamp()

        # Calculate monthly aggregates
        monthly_total = expenses.groupby("year_month")["amount"].sum().abs()
        monthly_nonessential = expenses[expenses["is_nonessential"]].groupby("year_month")["amount"].sum().abs()
        monthly_essential = expenses[~expenses["is_nonessential"]].groupby("year_month")["amount"].sum().abs()

        # Create complete date range
        if len(monthly_total) > 0:
            months_all = pd.period_range(
                tx["date"].min().to_period("M"),
                tx["date"].max().to_period("M"),
                freq="M"
            ).to_timestamp()
        else:
            raise ValueError("No monthly data available")
        
        monthly_total = monthly_total.reindex(months_all, fill_value=0.0)
        monthly_nonessential = monthly_nonessential.reindex(months_all, fill_value=0.0)
        monthly_essential = monthly_essential.reindex(months_all, fill_value=0.0)

        # Forecast next month
        next_month = (months_all.max() + pd.offsets.MonthBegin(1)).to_pydatetime()
        pred_nonessential_next = forecast_next_month(monthly_nonessential)

        # Current month analysis
        current_date = tx["date"].max()
        current_month_start = month_start(pd.Timestamp(current_date))
        mtd = expenses[(expenses["date"] >= current_month_start) & (expenses["date"] <= current_date)]
        mtd_nonessential = mtd[mtd["is_nonessential"]]["amount"].abs().sum()
        progress = month_progress(pd.Timestamp(current_date))
        pace_estimate = (mtd_nonessential / max(progress, 0.01)) if progress > 0 else 0.0

        # Top categories analysis (last 3 months) - for text output only
        last3_start = months_all.max() - pd.offsets.MonthBegin(2)
        last3 = expenses[expenses["is_nonessential"] & (expenses["date"] >= last3_start)].copy()
        
        if len(last3) > 0:
            top_cats = last3.groupby("category")["amount"].sum().abs().sort_values(ascending=False).head(5)
            top_cats_list = [(c, float(v)) for c, v in top_cats.items()]
        else:
            top_cats_list = []

        # Save monthly summary
        monthly_summary = pd.DataFrame({
            "month": months_all.strftime("%Y-%m"),
            "total_expenses": monthly_total.values,
            "essential_expenses": monthly_essential.values,
            "nonessential_expenses": monthly_nonessential.values
        })
        monthly_summary.to_csv(summary_csv_path, index=False)

        # Generate alerts
        alert_text = (
            f"Forecasted non-essential spend for {next_month.strftime('%B %Y')}: LKR {pred_nonessential_next:,.0f}\n"
            f"Month-to-date non-essential spend ({current_date.strftime('%B %Y')}): LKR {mtd_nonessential:,.0f}\n"
            f"Month progress: {progress*100:.1f}%\n"
            f"Pace-implied end-of-month non-essential spend: LKR {pace_estimate:,.0f}\n"
        )
        
        if pace_estimate > pred_nonessential_next * 1.1:
            alert_text += "⚠️  WARNING: On track to overspend non-essential budget by >10%.\n"
        else:
            alert_text += "✅ On track with spending goals.\n"
        
        if top_cats_list:
            alert_text += "\nTop non-essential categories to consider reducing (last 3 months):\n"
            for name, val in top_cats_list:
                alert_text += f"• {name}: LKR {val:,.0f}\n"
        else:
            alert_text += "\nNo significant non-essential spending in the last 3 months.\n"

        # Save alerts and forecast
        with open(alerts_path, "w", encoding="utf-8") as f:
            f.write(alert_text)

        forecast_data = {
            "next_month": next_month.strftime("%Y-%m"),
            "forecast_nonessential": round(float(pred_nonessential_next), 2),
            "pace_estimate_current_month": round(float(pace_estimate), 2),
            "current_month": current_date.strftime("%Y-%m"),
            "top_cats_last3": [{"category": c, "amount": round(v, 2)} for c, v in top_cats_list]
        }
        
        with open(forecast_json_path, "w", encoding="utf-8") as f:
            json.dump(forecast_data, f, indent=2)

        # Create only the forecast plot
        plt.style.use('default')
        
        # Forecast Plot - optimized size for GUI display
        fig_forecast, ax_forecast = plt.subplots(figsize=(10, 6), dpi=100)
        if len(monthly_nonessential) > 0:
            ax_forecast.plot(monthly_nonessential.index, monthly_nonessential.values, 
                           marker="o", color="#2E86AB", linewidth=2.5, markersize=6, 
                           label="Non-essential (actual)")
            ax_forecast.scatter([pd.Timestamp(next_month.year, next_month.month, 1)], 
                              [pred_nonessential_next], marker="x", color="#F24236", 
                              s=100, label="Forecast next", linewidths=3)
        
        ax_forecast.set_title("Non-essential Spending Forecast", fontsize=14, weight='bold', pad=20)
        ax_forecast.set_xlabel("Month", fontsize=12)
        ax_forecast.set_ylabel("Amount (LKR)", fontsize=12)
        ax_forecast.legend(fontsize=11)
        ax_forecast.grid(True, linestyle='--', alpha=0.5)
        ax_forecast.tick_params(axis='both', labelsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout(pad=2.0)
        plt.savefig(plot_forecast_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

        return alert_text, plot_forecast_path, monthly_summary
        
    except Exception as e:
        raise Exception(f"Error processing CSV: {str(e)}")

class ExpensePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personal Expense Predictor & Analyzer")
        
        # Start maximized and center on screen
        self.root.state('zoomed')  # Windows
        try:
            self.root.attributes('-zoomed', True)  # Linux
        except:
            pass
        
        # For macOS or fallback - get screen dimensions and maximize
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        self.root.resizable(True, True)
        
        # Set minimum window size
        self.root.minsize(1000, 700)

        # Configure enhanced styles for larger screens
        style = ttk.Style()
        style.configure("TButton", padding=10, font=("Arial", 12))
        style.configure("TLabel", font=("Arial", 11))
        style.configure("Header.TLabel", font=("Arial", 20, "bold"))
        style.configure("Subheader.TLabel", font=("Arial", 14, "bold"))

        # Create main container that fills the entire window
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Main content frame
        self.main_frame = ttk.Frame(self.main_container)
        self.main_frame.pack(fill="both", expand=True)

        # Header section
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill="x", pady=(0, 20))
        
        self.header_label = ttk.Label(header_frame, text="💰 Personal Expense Predictor & Analyzer", 
                                     style="Header.TLabel")
        self.header_label.pack()

        # Instructions
        self.instructions = ttk.Label(header_frame, 
                                     text="Upload a CSV file with columns: date, description, type, amount",
                                     font=("Arial", 12, "italic"))
        self.instructions.pack(pady=(5, 0))

        # Upload section
        self.upload_frame = ttk.LabelFrame(self.main_frame, text="📁 File Upload", padding=15)
        self.upload_frame.pack(fill="x", pady=(0, 15))
        
        upload_content = ttk.Frame(self.upload_frame)
        upload_content.pack(fill="x")
        
        self.upload_btn = ttk.Button(upload_content, text="📁 Select CSV File", 
                                    command=self.upload_file, style="TButton", width=20)
        self.upload_btn.pack(side="left")
        
        self.file_label = ttk.Label(upload_content, text="No file selected", 
                                   foreground="gray", font=("Arial", 11))
        self.file_label.pack(side="left", padx=(20, 0))

        # Create main content area - plot gets most of the space
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill="both", expand=True)
        
        # Left column - Results and Summary (small space)
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side="left", fill="y", padx=(0, 10))
        
        # Right column - Plot (gets most of the space)
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # Results section (left column) - compact
        self.results_frame = ttk.LabelFrame(left_frame, text="📊 Results", padding=10)
        self.results_frame.pack(fill="x", pady=(0, 10))
        
        self.results_text = scrolledtext.ScrolledText(self.results_frame, height=6, width=40,
                                                     font=("Consolas", 9), wrap=tk.WORD)
        self.results_text.pack(fill="both", expand=True)

        # Summary table section (left column) - compact
        self.summary_frame = ttk.LabelFrame(left_frame, text="📋 Summary", padding=10)
        self.summary_frame.pack(fill="both", expand=True)
        
        # Create compact treeview
        self.tree = ttk.Treeview(self.summary_frame, columns=("Month", "Total", "Essential", "Nonessential"), 
                                show="headings", height=8)
        
        # Configure column headings and compact widths
        self.tree.heading("Month", text="Month")
        self.tree.heading("Total", text="Total")
        self.tree.heading("Essential", text="Essential")
        self.tree.heading("Nonessential", text="Non-essential")
        
        self.tree.column("Month", width=80, anchor="center")
        self.tree.column("Total", width=100, anchor="e")
        self.tree.column("Essential", width=100, anchor="e")
        self.tree.column("Nonessential", width=120, anchor="e")
        
        # Add vertical scrollbar only for the compact table
        tree_scrollbar = ttk.Scrollbar(self.summary_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scrollbar.pack(side="right", fill="y")

        # Plot section (right column) - only forecast plot now
        self.plot_frame = ttk.LabelFrame(right_frame, text="📈 Spending Forecast", padding=15)
        self.plot_frame.pack(fill="both", expand=True)

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Selected: {filename}", foreground="blue")
            
            try:
                # Show processing message
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Processing file, please wait...")
                self.root.update()
                
                # Process the file (no categories plot returned)
                alert_text, plot_forecast_path, monthly_summary = process_csv(file_path)
                
                # Update results text
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, alert_text)

                # Update summary table
                for item in self.tree.get_children():
                    self.tree.delete(item)
                
                for _, row in monthly_summary.iterrows():
                    self.tree.insert("", tk.END, values=(
                        row["month"],
                        f"LKR {row['total_expenses']:,.0f}",
                        f"LKR {row['essential_expenses']:,.0f}",
                        f"LKR {row['nonessential_expenses']:,.0f}"
                    ))

                # Display only the forecast plot
                self._display_plot(plot_forecast_path, self.plot_frame)
                
                # Show success message
                messagebox.showinfo("Success", f"Analysis completed successfully!\nResults saved to ./outputs/")
                
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}\n\nPlease check that your CSV file has the required columns:\n• date\n• description\n• type\n• amount"
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, error_msg)
                messagebox.showerror("Error", f"Failed to process file:\n{str(e)}")

    def _display_plot(self, plot_path, frame):
        """Helper method to display forecast plot - static and fitted to frame"""
        try:
            # Clear existing content
            for widget in frame.winfo_children():
                widget.destroy()
            
            # Load image and resize to fit the frame perfectly
            img = Image.open(plot_path)
            
            # Get the frame dimensions (approximate)
            frame.update_idletasks()
            frame_width = max(frame.winfo_width() - 20, 400)  # Leave some padding
            frame_height = max(frame.winfo_height() - 20, 300)  # Leave some padding
            
            # Resize image to fit frame while maintaining aspect ratio
            img_ratio = img.width / img.height
            frame_ratio = frame_width / frame_height
            
            if img_ratio > frame_ratio:
                # Image is wider, fit to width
                new_width = frame_width
                new_height = int(frame_width / img_ratio)
            else:
                # Image is taller, fit to height
                new_height = frame_height
                new_width = int(frame_height * img_ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Create a simple label to display the plot
            plot_label = tk.Label(frame, image=photo, bg='white')
            plot_label.image = photo  # Keep a reference
            plot_label.pack(expand=True)  # Center the plot
            
        except Exception as e:
            # Show error message in frame
            error_label = ttk.Label(frame, text=f"Error loading plot:\n{str(e)}", 
                                  justify=tk.CENTER, font=("Arial", 12))
            error_label.pack(fill="both", expand=True)

def main():
    root = tk.Tk()
    app = ExpensePredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
