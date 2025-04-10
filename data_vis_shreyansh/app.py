import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for server environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from io import BytesIO
from flask import Flask, request, send_file
from datetime import datetime
import squarify
import seaborn as sns
import pandas as pd


# Print available styles (for debugging)
available_styles = plt.style.available
print("Available styles:", available_styles)

# Use the exact style name as it appears in your list
plt.style.use('seaborn-v0_8-colorblind')


app = Flask(__name__)

def load_data(filename):
    """Utility function to load JSON data from a file."""
    with open(filename, 'r') as f:
        return json.load(f)

def parse_date(date_str):
    """Parse a date string (format: YYYY-MM-DD) to a datetime object."""
    return datetime.strptime(date_str, "%Y-%m-%d")

def filter_customer_data(data, region=None, loyalty=None, start_date=None, end_date=None):
    """Filter customer data based on provided query parameters.
       For date filtering, records missing 'signup date' are skipped.
    """
    filtered = data
    if region:
        filtered = [d for d in filtered if d.get("region", "").lower() == region.lower()]
    if loyalty:
        # Make sure to check the correct key name for loyalty.
        # If your dataset has "loyalty tier", use that; otherwise adjust accordingly.
        filtered = [d for d in filtered if d.get("loyalty", "").lower() == loyalty.lower()]
    if start_date:
        start = parse_date(start_date)
        filtered = [d for d in filtered if d.get("signup date") and parse_date(d.get("signup date")) >= start]
    if end_date:
        end = parse_date(end_date)
        filtered = [d for d in filtered if d.get("signup date") and parse_date(d.get("signup date")) <= end]
    return filtered
# ---------------------------------------------------------------------
# 1) Customer Growth Over Time (Line Chart)
# Endpoint: /api/customer/growth
# ---------------------------------------------------------------------
@app.route('/api/customer/growth', methods=['GET'])
def get_customer_growth_chart():
    data = load_data('customer_data.json')
    region = request.args.get('region')
    loyalty = request.args.get('loyalty')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    filtered = filter_customer_data(data, region, loyalty, start_date, end_date)

    # Group sign-ups by month (YYYY-MM); skip records missing "signup date"
    monthly_counts = {}
    for record in filtered:
        signup_date = record.get("signup date")
        if not signup_date:
            continue
        try:
            dt = parse_date(signup_date)
        except Exception:
            continue
        key = dt.strftime("%Y-%m")
        monthly_counts[key] = monthly_counts.get(key, 0) + 1

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    if monthly_counts:
        # Sort keys to ensure x-axis is in chronological order
        sorted_keys = sorted(monthly_counts.keys())
        counts = [monthly_counts[k] for k in sorted_keys]

        # Convert x-axis labels to numeric indices
        x_vals = np.arange(len(sorted_keys))

        # Plot data at these numeric positions
        ax.plot(x_vals, counts, marker='o', linestyle='-', color='blue', label='New Sign-Ups')

        # Set x-axis tick positions and labels
        ax.set_xticks(x_vals)
        ax.set_xticklabels(sorted_keys, rotation=45, ha='right')

        # Limit the number of x-ticks to avoid overlap
        ax.xaxis.set_major_locator(ticker.MaxNLocator(12))

        # Labeling & Title
        ax.set_title("Monthly Customer Growth Over Time", fontsize=16, fontweight='bold')
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("New Sign-Ups", fontsize=12)
        ax.legend(loc='upper left')

        # Adjust bottom margin so labels aren't cut off
        plt.subplots_adjust(bottom=0.2)
    else:
        # If no data is available
        ax.text(0.5, 0.5, "No signup date data available",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Monthly Customer Growth Over Time")

    # Ensure all elements fit nicely
    plt.tight_layout()

    # Save chart to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)

    return send_file(buffer, mimetype='image/png')
# ---------------------------------------------------------------------
# 2) Customer Segmentation by Loyalty (Pie Chart)
# Endpoint: /api/customer/segmentation
# ---------------------------------------------------------------------
@app.route('/api/customer/segmentation', methods=['GET'])


def get_customer_segmentation_chart():
    data = load_data('customer_data.json')
    region = request.args.get('region')
    # Filter data by region if provided
    filtered = data if not region else [d for d in data if d.get("region", "").lower() == region.lower()]
    
    # Count customers by loyalty tier using the correct field name "loyalty tier"
    loyalty_counts = {}
    for record in filtered:
        tier = record.get("loyalty tier", "Unknown")
        loyalty_counts[tier] = loyalty_counts.get(tier, 0) + 1

    labels = list(loyalty_counts.keys())
    sizes = list(loyalty_counts.values())

    plt.figure(figsize=(8, 8), dpi=100)
    if sizes and sum(sizes) > 0:
        # Add a slight explosion for each slice for visual effect
        explode = [0.05] * len(labels)
        # Use a colormap to assign colors to each slice
        colors = plt.cm.Paired(range(len(labels)))
        
        wedges, texts, autotexts = plt.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%', 
            startangle=90, 
            explode=explode, 
            colors=colors,
            shadow=True
        )
        plt.title("Customer Segmentation by Loyalty Tier", fontsize=16, fontweight='bold')
        plt.axis('equal')  # ensures the pie is circular

        # Improve legibility of text labels
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(12)
    else:
        plt.text(0.5, 0.5, "No data available",
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.title("Customer Segmentation by Loyalty Tier", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save the figure to a bytes buffer and return it as a PNG
    from io import BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return send_file(buffer, mimetype='image/png')

# ---------------------------------------------------------------------
# 3) Customer Churn Rate (Bar Chart & Trend Line)
# Endpoint: /api/customer/churn
# ---------------------------------------------------------------------
@app.route('/api/customer/churn', methods=['GET'])
def get_customer_churn_chart():
    data = load_data('customer_data.json')
    region = request.args.get('region')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    filtered = filter_customer_data(data, region=region, start_date=start_date, end_date=end_date)

    # Define churn logic
    churned = [rec for rec in filtered if rec.get("status", "").lower() == "inactive"]
    retained = [rec for rec in filtered if rec.get("status", "").lower() == "active"]

    # Prepare the figure with 1 row and 2 columns
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    fig.subplots_adjust(wspace=0.25)  # Adjust horizontal space between subplots

    # -----------------------------
    # Left Subplot: Churned vs. Retained (Bar Chart)
    # -----------------------------
    bar_labels = ["Churned", "Retained"]
    bar_values = [len(churned), len(retained)]
    colors = ['#d62728', '#2ca02c']  # Red & Green in a friendlier color palette

    # Create the bar chart
    bars = ax[0].bar(bar_labels, bar_values, color=colors, alpha=0.8)
    ax[0].set_title("Churned vs. Retained Customers", fontsize=14, fontweight='bold')
    ax[0].set_ylabel("Number of Customers", fontsize=12)
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Add numeric labels above each bar
    for bar in bars:
        height = bar.get_height()
        ax[0].annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),  # Offset text 5 points above top of the bar
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')

    # -----------------------------
    # Right Subplot: Monthly Churn Trend (Line Chart)
    # -----------------------------
    churn_monthly = {}
    for rec in churned:
        signup_date = rec.get("signup date")
        if not signup_date:
            continue
        try:
            dt = parse_date(signup_date)
        except Exception:
            continue
        key = dt.strftime("%Y-%m")
        churn_monthly[key] = churn_monthly.get(key, 0) + 1

    ax[1].set_title("Monthly Churn Trend", fontsize=14, fontweight='bold')
    ax[1].set_xlabel("Month", fontsize=12)
    ax[1].set_ylabel("Churned Customers", fontsize=12)
    ax[1].grid(linestyle='--', alpha=0.7)

    if churn_monthly:
        # Sort the months chronologically
        sorted_months = sorted(churn_monthly.keys())
        churn_counts = [churn_monthly[m] for m in sorted_months]

        # Convert to numeric x positions
        x_vals = np.arange(len(sorted_months))
        ax[1].plot(x_vals, churn_counts, marker='o', linestyle='-', color='#d62728', alpha=0.9)

        # Set x ticks and labels
        ax[1].set_xticks(x_vals)
        ax[1].set_xticklabels(sorted_months, rotation=45, ha='right')
        # Limit max number of ticks if needed
        ax[1].xaxis.set_major_locator(ticker.MaxNLocator(12))
    else:
        ax[1].text(0.5, 0.5, "No signup date data available",
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax[1].transAxes, fontsize=12)

    plt.tight_layout()

    # Save the figure to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)

    return send_file(buffer, mimetype='image/png')

# ---------------------------------------------------------------------
# Other Endpoints Remain Unchanged
# ---------------------------------------------------------------------
def load_data(filename):
    """Load JSON data from the given filename."""
    with open(filename, 'r') as f:
        return json.load(f)

def filter_product_data(data, category=None):
    """Filter product data by category, if given."""
    if category:
        return [d for d in data if d.get("category", "").lower() == category.lower()]
    return data

# Apply a modern style
plt.style.use('seaborn-v0_8-colorblind')

# ---------------------------------------------------------------------
# 1) Top-Selling Products & Categories (Horizontal Bar Chart)
#    Endpoint: /api/product/topsellers
# ---------------------------------------------------------------------
@app.route('/api/product/topsellers', methods=['GET'])
def get_product_topsellers_chart():
    """
    Returns an attractive horizontal bar chart ranking products by revenue.
    Revenue is calculated as: price * units_solds.
    Optional filter: ?category=XYZ.
    """
    data = load_data('product_data.json')
    category_filter = request.args.get('category')
    filtered = filter_product_data(data, category_filter)
    
    # Calculate revenue using the fields "price" and "units_solds"
    product_sales = []
    for item in filtered:
        try:
            price = float(item.get("price", 0))
        except ValueError:
            price = 0
        try:
            units = float(item.get("units_solds", 0))  # Note: using "units_solds"
        except ValueError:
            units = 0
        revenue = price * units
        product_sales.append((item.get("name", "Unknown"), revenue))
    
    # Sort by revenue descending and choose top 10
    product_sales.sort(key=lambda x: x[1], reverse=True)
    top_n = product_sales[:10]
    names = [t[0] for t in top_n]
    revenues = [t[1] for t in top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    if revenues and sum(revenues) > 0:
        y_positions = np.arange(len(names))
        bars = ax.barh(y_positions, revenues, color='mediumseagreen', edgecolor='black')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(names, fontsize=12)
        ax.invert_yaxis()  # Highest revenue at the top
        
        ax.set_xlabel("Revenue", fontsize=12)
        title_text = "Top-Selling Products" + (f" in {category_filter.title()}" if category_filter else "")
        ax.set_title(title_text, fontsize=16, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
    
        # Annotate each bar with the revenue value
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + (0.01 * max(revenues)), bar.get_y() + bar.get_height() / 2,
                    f"${width:,.2f}", va='center', ha='left', fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "No sales data available", ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Top-Selling Products", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')


# ---------------------------------------------------------------------
# 2) Stock Levels & Inventory Turnover (Vertical Bar Chart)
#    Endpoint: /api/product/stock
# ---------------------------------------------------------------------
@app.route('/api/product/stock', methods=['GET'])
def get_product_stock_chart():
    """
    Returns a colorful vertical bar chart showing each product's stock levels
    with significantly increased spacing between bars for clarity.
    You can limit the number of products shown by ?limit=NUMBER (default 30).
    Optionally pass a threshold (?threshold=50) to highlight bars below that in red.
    Filter by category (?category=XYZ).
    """
    data = load_data('product_data.json')
    category_filter = request.args.get('category')
    threshold_str = request.args.get('threshold')
    limit_str = request.args.get('limit')   # e.g., ?limit=20

    # Convert threshold and limit to float/int if provided
    threshold = None
    if threshold_str:
        try:
            threshold = float(threshold_str)
        except ValueError:
            threshold = None

    limit_products = 30  # Default limit
    if limit_str:
        try:
            limit_products = int(limit_str)
        except ValueError:
            pass  # Keep default if conversion fails

    # Filter by category if provided
    filtered = filter_product_data(data, category_filter)

    # Sort by descending stock
    filtered.sort(key=lambda x: float(x.get("stock levels", 0)), reverse=True)
    
    # Limit to top N for readability
    filtered = filtered[:limit_products]

    # Extract names and stock data
    names = [item.get("name", "Unknown") for item in filtered]
    stocks = [float(item.get("stock levels", 0)) for item in filtered]

    # Increase figure size to allow more horizontal space
    fig, ax = plt.subplots(figsize=(16, 8), dpi=120)
    
    # Adjust margins to avoid overlap with title/labels
    plt.subplots_adjust(top=0.88, bottom=0.30)

    # If there's valid stock data, create a bar chart
    if len(stocks) > 0 and any(s > 0 for s in stocks):
        import matplotlib.colors as mcolors

        # We'll space out the bars by multiplying the x positions
        x_spacing = 1.5  # Increase this for more spacing
        x_positions = np.arange(len(names)) * x_spacing

        # Determine min/max stock for color normalization
        lowest_stock = min(stocks)
        highest_stock = max(stocks)
        norm = mcolors.Normalize(vmin=lowest_stock, vmax=highest_stock)
        cmap = plt.cm.Spectral

        # Create color array for each bar
        bar_colors = [cmap(norm(s)) for s in stocks]
        # Decrease the bar width to further separate bars
        bar_width = 0.9

        bars = ax.bar(x_positions, stocks, color=bar_colors, edgecolor='black', width=bar_width)

        # Highlight bars below threshold
        if threshold is not None:
            for i, bar in enumerate(bars):
                if stocks[i] < threshold:
                    bar.set_color('crimson')

        # X-axis labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)

        # Y-axis label
        ax.set_ylabel("Stock Levels", fontsize=12)

        # Chart title
        title_text = f"Top {limit_products} Product Stock Levels"
        if category_filter:
            title_text += f" in {category_filter.title()}"
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)

        # Add a subtle grid
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # Annotate each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + (0.01 * highest_stock),
                    f"{int(height)}",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    else:
        # No valid data
        ax.text(0.5, 0.5, "No stock data available", ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Product Stock Levels", fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')


# ---------------------------------------------------------------------
# 3) Pricing Analysis (Scatter Plot)
#    Endpoint: /api/product/pricing
# ---------------------------------------------------------------------
@app.route('/api/product/pricing', methods=['GET'])
def get_product_pricing_chart():
    """
    Returns an attractive scatter plot showing product price versus units sold.
    This helps identify pricing trends relative to sales performance.
    Optional: filter by category (?category=XYZ).
    """
    data = load_data('product_data.json')
    category_filter = request.args.get('category')
    filtered = filter_product_data(data, category_filter)
    
    prices = []
    volumes = []
    labels = []
    
    for item in filtered:
        try:
            p = float(item.get("price", 0))
        except ValueError:
            p = 0
        try:
            # Use the field "units_solds" since your JSON has that key.
            v = float(item.get("units_solds", 0))
        except ValueError:
            v = 0
        prices.append(p)
        volumes.append(v)
        labels.append(item.get("name", "Unknown"))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    if prices and sum(volumes) > 0:
        scatter = ax.scatter(prices, volumes, c=prices, cmap='viridis', alpha=0.8, edgecolor='black')
        ax.set_xlabel("Price", fontsize=12)
        ax.set_ylabel("Units Sold", fontsize=12)
        title_text = "Pricing Analysis" + (f" in {category_filter.title()}" if category_filter else "")
        ax.set_title(title_text, fontsize=16, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.ax.set_ylabel('Price', fontsize=12)
        ax.grid(linestyle='--', alpha=0.6)
        # Optionally, annotate only if there are few points
        if len(prices) <= 20:
            for i, label in enumerate(labels):
                ax.annotate(label, (prices[i], volumes[i]), fontsize=8, alpha=0.8)
    else:
        ax.text(0.5, 0.5, "No pricing or sales data available", ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Pricing Analysis", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

def load_data(filename):
    """Load JSON data from the given filename."""
    with open(filename, 'r') as f:
        data = json.load(f)
    # Debug: report number of records loaded
    print("[DEBUG] Loaded", len(data), "records from", filename)
    return data

def parse_date(date_str):
    """Parse a date string (expected format YYYY-MM-DD) into a datetime object."""
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except Exception as e:
        print(f"[DEBUG] Error parsing date '{date_str}':", e)
        return None

# Use a pleasing style (using a seaborn style available in your environment)
try:
    plt.style.use('seaborn-v0_8-colorblind')
except Exception as e:
    print(e)
    plt.style.use('seaborn-v0_8-darkgrid')

# ---------------------------------------------------------------------
# 1) Order Volume and Revenue Trend (Line Chart)
# Endpoint: /api/order/trend
# ---------------------------------------------------------------------
@app.route('/api/order/trend', methods=['GET'])
def order_trend_chart():
    data = load_data('order_data.json')
    
    monthly_orders = {}
    monthly_revenue = {}

    for idx, record in enumerate(data):
        order_date = record.get("order date")
        if not order_date:
            print(f"[DEBUG] Record #{idx} missing 'order date'")
            continue
        dt = parse_date(order_date)
        if dt is None:
            print(f"[DEBUG] Record #{idx} has unparseable date: '{order_date}'")
            continue
        month_key = dt.strftime("%Y-%m")
        try:
            revenue = float(record.get("total_amount", 0))
        except Exception as e:
            print(f"[DEBUG] Error converting total_amount in record #{idx}: {record.get('total_amount')}")
            revenue = 0.0
        monthly_orders[month_key] = monthly_orders.get(month_key, 0) + 1
        monthly_revenue[month_key] = monthly_revenue.get(month_key, 0) + revenue

    print("[DEBUG] Monthly Orders:", monthly_orders)
    print("[DEBUG] Monthly Revenue:", monthly_revenue)

    sorted_months = sorted(monthly_orders.keys())
    if not sorted_months:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No order date data available",
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Order Volume and Revenue Trend", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    orders_data = [monthly_orders[m] for m in sorted_months]
    revenue_data = [monthly_revenue[m] for m in sorted_months]
    
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=120)
    plt.subplots_adjust(top=0.88, bottom=0.2)
    x_vals = np.arange(len(sorted_months))
    
    ax1.plot(x_vals, orders_data, marker='o', linestyle='-', color='blue', label='Orders')
    ax1.set_xlabel("Month", fontsize=12)
    ax1.set_ylabel("Number of Orders", fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(x_vals)
    ax1.set_xticklabels(sorted_months, rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    ax2.plot(x_vals, revenue_data, marker='s', linestyle='--', color='green', label='Revenue')
    ax2.set_ylabel("Total Revenue ($)", fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(12))
    ax1.set_title("Order Volume and Revenue Trend", fontsize=14, fontweight='bold', pad=15)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')


# ---------------------------------------------------------------------
# 2) Order Status Tracking (Stacked Bar Chart)
# Endpoint: /api/order/status
# ---------------------------------------------------------------------
@app.route('/api/order/status', methods=['GET'])
def order_status_chart():
    from io import BytesIO
    import matplotlib.pyplot as plt
    import numpy as np

    data = load_data('order_data.json')
    print(f"[DEBUG] Loaded {len(data)} records from order_data.json for status tracking.")

    monthly_status_counts = {}

    # Loop through each record and attempt to parse date + status
    for idx, record in enumerate(data):
        order_date = record.get("order date")
        status = record.get("shipping_status", "Unknown")
        
        # Debug info
        print(f"\n[DEBUG] Record #{idx+1}:")
        print("   order_date:", order_date)
        print("   shipping_status:", status)

        if not order_date:
            print(f"[DEBUG]  -> Skipping record because 'order date' is missing.")
            continue

        dt = parse_date(order_date)  # parse_date strips whitespace and expects YYYY-MM-DD
        if dt is None:
            print(f"[DEBUG]  -> Could not parse date: '{order_date}'. Skipping.")
            continue

        month_key = dt.strftime("%Y-%m")
        print(f"[DEBUG]  -> month_key: {month_key}, shipping_status: {status}")

        # Initialize the dictionary for that month if needed
        if month_key not in monthly_status_counts:
            monthly_status_counts[month_key] = {}
        
        # Increment the count for this status
        monthly_status_counts[month_key][status] = \
            monthly_status_counts[month_key].get(status, 0) + 1

    # Show the final grouped data
    print("[DEBUG] Final monthly_status_counts:", monthly_status_counts)

    # Collect all statuses to make the stacked bar segments
    all_statuses = set()
    for stats_dict in monthly_status_counts.values():
        all_statuses.update(stats_dict.keys())
    all_statuses = sorted(list(all_statuses))

    # Sort the months
    sorted_months = sorted(monthly_status_counts.keys())
    print("[DEBUG] Sorted Months:", sorted_months)

    # If no months found, show the 'no data' chart
    if not sorted_months:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No order date or shipping status data available",
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Order Status Tracking", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    x_vals = np.arange(len(sorted_months))
    bottom_stack = np.zeros(len(sorted_months))

    # Create a large figure for clarity
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    plt.subplots_adjust(right=0.78, top=0.88, bottom=0.2)

    # Build stacked bars for each status
    for status in all_statuses:
        bar_vals = []
        for month in sorted_months:
            # If this status doesn't appear in that month, use 0
            count = monthly_status_counts[month].get(status, 0)
            bar_vals.append(count)
        bar_vals = np.array(bar_vals)
        ax.bar(x_vals, bar_vals, bottom=bottom_stack, label=status)
        bottom_stack += bar_vals

    # Configure x-axis
    ax.set_xticks(x_vals)
    ax.set_xticklabels(sorted_months, rotation=45, ha='right', fontsize=10)

    # Configure labels and title
    ax.set_ylabel("Number of Orders", fontsize=12)
    ax.set_title("Order Status Tracking by Month", fontsize=14, fontweight='bold', pad=15)
    ax.legend(title="Shipping Status", loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')


# ---------------------------------------------------------------------
# 3) Average Order Value and Conversion Rate (Gauge & Funnel)
# Endpoint: /api/order/value
# ---------------------------------------------------------------------
@app.route('/api/order/value', methods=['GET'])
def order_value_chart():
    data = load_data('order_data.json')
    total_orders = 0
    total_revenue = 0.0
    for record in data:
        total_orders += 1
        try:
            total_revenue += float(record.get("total_amount", 0))
        except ValueError:
            total_revenue += 0.0

    avg_order_value = total_revenue / total_orders if total_orders else 0.0

    try:
        target = float(request.args.get('target', 100.0))
    except ValueError:
        target = 100.0

    try:
        visits = int(request.args.get('visits', 10000))
    except ValueError:
        visits = 10000

    delivered_count = sum(1 for r in data if r.get("shipping_status", "").lower() == "delivered")

    fig = plt.figure(figsize=(14, 6), dpi=120)
    plt.subplots_adjust(top=0.85, wspace=0.3)

    # Subplot 1: Gauge-like view for Average Order Value
    ax1 = fig.add_subplot(1, 2, 1)
    gauge_width = 0.5
    max_scale = max(avg_order_value, target) * 1.2 or 100.0

    ax1.barh(["AOV"], [max_scale], color='lightgray', alpha=0.7, height=gauge_width)
    ax1.barh(["AOV"], [target], color='orange', alpha=0.8, height=gauge_width, label="Target")
    ax1.barh(["AOV"], [avg_order_value], color='green', alpha=0.8, height=gauge_width, label="Actual AOV")
    
    ax1.set_xlim(0, max_scale)
    ax1.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax1.spines[spine].set_visible(False)
    ax1.set_title("Average Order Value", fontsize=12, fontweight='bold', pad=15)
    ax1.legend(loc='lower right')
    ax1.text(avg_order_value + 1, 0, f"${avg_order_value:,.2f}", va='center', fontsize=11, fontweight='bold', color='black')
    ax1.text(target + 1, 0.35, f"Target: ${target:,.2f}", va='center', fontsize=10, fontweight='bold', color='black')

    # Subplot 2: Conversion Funnel
    ax2 = fig.add_subplot(1, 2, 2)
    funnel_labels = ["Visits", "Orders", "Delivered"]
    funnel_values = [visits, total_orders, delivered_count]
    x_positions = np.arange(len(funnel_labels)) * 2.0  # space them out
    max_width = max(funnel_values) * 1.1

    for i, (label, val) in enumerate(zip(funnel_labels, funnel_values)):
        y_pos = len(funnel_labels) - i - 1  # Invert order so top is Visits
        ax2.barh(y_pos, val, color='royalblue', alpha=0.8)
        ax2.text(val + (0.02 * max_width), y_pos, f"{val}", va='center', ha='left', fontsize=10, fontweight='bold')

    ax2.set_yticks([2, 1, 0])
    ax2.set_yticklabels(["Visits", "Orders", "Delivered"], fontsize=11)
    ax2.set_xlim(0, max_width)
    ax2.set_ylim(-0.5, 2.5)
    ax2.invert_yaxis()
    ax2.set_title("Conversion Funnel", fontsize=12, fontweight='bold', pad=15)
    for spine in ["top", "right", "left", "bottom"]:
        ax2.spines[spine].set_visible(False)
    ax2.set_xticks([])

    fig.suptitle("Average Order Value & Conversion Rate", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

def load_data(filename):
    """Load JSON data from the given filename."""
    with open(filename, 'r') as f:
        return json.load(f)

# =============================================================================
# Endpoint 1: Payment Terms Distribution Pie Chart
# =============================================================================
def load_data(filename):
    """Load JSON data from the given filename."""
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"[DEBUG] Loaded {len(data)} supplier records from {filename}")
    return data

# If you need date parsing, define parse_date here (not strictly needed for suppliers).
def parse_date(date_str):
    """Parse date string if used. Currently unused unless you store dates for suppliers."""
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except:
        return None

# Try using a colorful style
try:
    plt.style.use('seaborn-v0_8-colorblind')
except:
    plt.style.use('seaborn-v0_8-darkgrid')


# ---------------------------------------------------------------------
# 1) Supplier Performance and Delivery Times
#     Endpoint: /api/supplier/performance
#
#   Visualizations:
#     - Bullet/Bar Chart: Compare actual delivery times vs. SLA_days
#     - Scatter Plot: delivery_time_days vs. quality_rating
# ---------------------------------------------------------------------
@app.route('/api/supplier/performance', methods=['GET'])
def supplier_performance_chart():
    """
    Refined Supplier Performance & Delivery Times:
      - Left: Bullet-style horizontal bar chart for top 20 suppliers by delivery_time_days.
        Compares 'delivery_time_days' vs. 'SLA_days'. Thinner bars, smaller fonts, no numeric annotations.
      - Right: Scatter plot for all valid records, showing 'delivery_time_days' vs. 'quality_rating'
        with smaller markers, partial transparency, optional color scale.
    """
    data = load_data('supplier_data.json')

    # We'll assume each record has: 'delivery_time_days', 'SLA_days', 'quality_rating'
    valid_records = []
    for rec in data:
        if ("delivery_time_days" in rec and
            "SLA_days" in rec and
            "quality_rating" in rec):
            valid_records.append(rec)

    if not valid_records:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No performance data available",
                ha='center', va='center', fontsize=14)
        ax.set_title("Supplier Performance & Delivery Times", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    # -------------------------------------------------------------
    # 1) Bullet Chart for top 20 suppliers by 'delivery_time_days'
    # -------------------------------------------------------------
    # Sort valid records by 'delivery_time_days' descending, then take top 20
    top_n = 20
    sorted_by_delivery = sorted(valid_records, key=lambda r: r["delivery_time_days"], reverse=True)
    bullet_records = sorted_by_delivery[:top_n]

    # Prepare data
    suppliers_bullet = [f"Sup {r['Supplier ID']}" for r in bullet_records]
    delivery_times = [r["delivery_time_days"] for r in bullet_records]
    sla_times = [r["SLA_days"] for r in bullet_records]

    # We'll create x_positions from 0..(top_n-1)
    x_positions = np.arange(len(bullet_records))

    # -------------------------------------------------------------
    # 2) Scatter Plot for all valid records
    # -------------------------------------------------------------
    # We'll just show everyone, no top-n limit
    lead_times = [r["delivery_time_days"] for r in valid_records]
    qualities = [r["quality_rating"] for r in valid_records]

    # -------------------------------------------------------------
    # Create the figure
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    plt.subplots_adjust(wspace=0.3)

    # -------------------------------------------------------------
    # Left Subplot: Bullet-Style Bar Chart
    # -------------------------------------------------------------
    ax1.set_title("Delivery Time vs. SLA", fontsize=12, fontweight='bold', pad=10)
    
    # Plot SLA as a grey background bar
    for i, sla_val in enumerate(sla_times):
        ax1.barh(y=i, width=sla_val, color='lightgray', alpha=0.6, height=0.8)

    # Plot actual delivery times with partial transparency
    ax1.barh(
        y=x_positions,
        width=delivery_times,
        color='blue',
        alpha=0.7,
        height=0.4,  # thinner bars
        label="Actual Delivery"
    )

    # Format the y-axis with smaller fonts, skipping numeric annotations
    ax1.set_yticks(x_positions)
    ax1.set_yticklabels(suppliers_bullet, fontsize=9)
    ax1.invert_yaxis()  # top=largest
    ax1.set_xlabel("Days", fontsize=10)
    ax1.legend(loc='lower right', fontsize=9)

    # -------------------------------------------------------------
    # Right Subplot: Scatter Plot (Lead Time vs. Quality)
    # -------------------------------------------------------------
    ax2.set_title("Lead Time vs. Quality Rating", fontsize=12, fontweight='bold', pad=10)

    # We'll color the markers by lead time (optional). 
    sc = ax2.scatter(
        lead_times,
        qualities,
        c=lead_times,
        cmap='coolwarm',
        alpha=0.6,       # partial transparency
        s=30,            # smaller marker size
        edgecolor='gray'
    )
    ax2.set_xlabel("Delivery Time (days)", fontsize=10)
    ax2.set_ylabel("Quality Rating", fontsize=10)
    ax2.grid(linestyle='--', alpha=0.5)

    # Optional colorbar
    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label("Lead Time (days)", fontsize=9)

    # -------------------------------------------------------------
    # Final Figure
    # -------------------------------------------------------------
    fig.suptitle("Supplier Performance & Delivery Times", fontsize=14, fontweight='bold', y=0.98)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')



# ---------------------------------------------------------------------
# 2) Spend Analysis by Supplier
#     Endpoint: /api/supplier/spend
#
#   Visualizations:
#     - Treemap: visualize 'spend_amount' distribution across suppliers
#     - Trend Line: track 'spend_amount' trends (e.g., sorted by Supplier ID)
# ---------------------------------------------------------------------
def load_data(filename):
    """Load JSON data from the given filename."""
    with open(filename, 'r') as f:
        return json.load(f)

# ---------------------------------------------------------------------
# Supplier Spend Analysis Endpoint (Treemap + Trend Line)
# Endpoint: /api/supplier/spend
# ---------------------------------------------------------------------
@app.route('/api/supplier/spend', methods=['GET'])
def supplier_spend_chart():
    """
    Returns a two-panel chart:
      - Left: A treemap showing spending distribution across the top 15 suppliers.
      - Right: A line chart showing the spending trend by Supplier ID.
    """
    data = load_data('supplier_data.json')

    # Filter valid supplier records that have a positive 'spend_amount'
    valid_records = [
        r for r in data
        if "spend_amount" in r and isinstance(r["spend_amount"], (int, float)) and r["spend_amount"] > 0
    ]

    if not valid_records:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No spend data available",
                ha='center', va='center', fontsize=14)
        ax.set_title("Supplier Spend Analysis", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    # Sort by spend_amount (descending) and take top 15 suppliers
    sorted_recs = sorted(valid_records, key=lambda r: r["spend_amount"], reverse=True)
    top_n = 15
    top_suppliers = sorted_recs[:top_n]

    # Prepare data for the treemap.
    # Each label is a shortened version: "Sup {ID}"
    treemap_labels = [f"Sup {r['Supplier ID']}" for r in top_suppliers]
    treemap_spends = [float(r["spend_amount"]) for r in top_suppliers]

    # For the trend line, sort the same top suppliers by Supplier ID.
    subset_sorted_by_id = sorted(top_suppliers, key=lambda r: r["Supplier ID"])
    ids_line = [r["Supplier ID"] for r in subset_sorted_by_id]
    spends_line = [float(r["spend_amount"]) for r in subset_sorted_by_id]

    # Create the figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    plt.subplots_adjust(wspace=0.3)

    # ----- Left Subplot: Treemap -----
    ax1.set_title("Spend Distribution (Treemap)", fontsize=13, fontweight='bold', pad=10)
    # Use a vibrant color palette
    num_categories = len(treemap_spends)
    cmap = sns.color_palette("Spectral", num_categories)
    
    # Plot treemap using squarify
    # Passing the ax parameter directs the treemap to the given subplot.
    squarify.plot(
        sizes=treemap_spends,
        label=treemap_labels,
        color=cmap,
        alpha=0.8,
        pad=True,
        text_kwargs={'fontsize': 9},
        ax=ax1
    )
    # Set axis limits to match squarify's coordinate system ([0,100])
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.invert_yaxis()  # So the origin is at the top-left
    ax1.axis('off')

    # ----- Right Subplot: Spending Trend (Line Chart) -----
    ax2.plot(
        ids_line,
        spends_line,
        marker='o',
        markersize=5,
        linestyle='-',
        color='purple',
        alpha=0.8
    )
    ax2.set_xlabel("Supplier ID (Top 15)", fontsize=11)
    ax2.set_ylabel("Spend Amount", fontsize=11)
    ax2.set_title("Spending Trend by Supplier ID", fontsize=13, fontweight='bold')
    ax2.grid(linestyle='--', alpha=0.6)

    fig.suptitle("Supplier Spend Analysis", fontsize=15, fontweight='bold', y=0.98)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')


# ---------------------------------------------------------------------
# 3) Supplier Risk Assessment
#     Endpoint: /api/supplier/risk
#
#   Visualization:
#     - Heatmap: Evaluate performance based on timeliness (delivery_time_days), quality, 
#       payment terms, risk_index, etc.
# ---------------------------------------------------------------------
@app.route('/api/supplier/risk', methods=['GET'])
def supplier_risk_chart():
    data = load_data('supplier_data.json')

    # We'll assume "delivery_time_days", "quality_rating", and "risk_index" exist
    # and possibly factor "payment terms" in if you define a numeric encoding for it.
    valid_records = []
    for r in data:
        if ("delivery_time_days" in r and
            "quality_rating" in r and
            "risk_index" in r):
            valid_records.append(r)

    if not valid_records:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No numeric fields for risk analysis", ha='center', va='center', fontsize=14)
        ax.set_title("Supplier Risk Assessment", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    import pandas as pd

    # Suppose we want a correlation among these three numeric fields:
    # [delivery_time_days, quality_rating, risk_index].
    # If you want to incorporate payment terms, you'd have to encode them (e.g. Net 30 -> 30).
    suppliers = [r["Supplier ID"] for r in valid_records]
    dtimes = [r["delivery_time_days"] for r in valid_records]
    quality = [r["quality_rating"] for r in valid_records]
    risk = [r["risk_index"] for r in valid_records]

    df = pd.DataFrame({
        "DeliveryTime": dtimes,
        "Quality": quality,
        "RiskIndex": risk
    }, index=suppliers)

    # We'll create a correlation heatmap + direct numeric heatmap
    corr = df.corr()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    plt.subplots_adjust(wspace=0.4)
    
    # Correlation heatmap
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax1, square=True)
    ax1.set_title("Correlation Heatmap", fontsize=12, fontweight='bold')

    # Numeric heatmap of raw data
    # If you have widely varying scales, consider normalizing or standardizing.
    sns.heatmap(df, cmap="YlGnBu", ax=ax2, cbar_kws={"shrink": 0.7}, annot=False)
    ax2.set_title("Supplier Metrics Heatmap", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Metrics")
    ax2.set_ylabel("Supplier ID")

    fig.suptitle("Supplier Risk Assessment", fontsize=14, fontweight='bold', y=0.98)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

def load_employee_data():
    with open("employee_data.json", "r") as f:
        data = json.load(f)
    return data

# ---------------------------------------------------------------------
# HELPER: Parse date strings like "2020-06-15" -> datetime.date
# ---------------------------------------------------------------------
def parse_date(date_str):
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except:
        return None

# ---------------------------------------------------------------------
# 1) Headcount Trend (Area Chart) & Turnover Rate by Department (Bar Chart)
#    /api/employee/headcount
# ---------------------------------------------------------------------
@app.route('/api/employee/headcount', methods=['GET'])
def employee_headcount_chart():
    """
    Improved Headcount (Area Chart) & Turnover (Bar Chart):
      - Larger figure, lighter fill color, reduced x-axis tick density
      - Clearer labeling and spacing
    """
    from datetime import datetime
    import matplotlib.dates as mdates

    data = load_employee_data()  # your function to load employee_data.json

    if not data:
        fig, ax = plt.subplots(figsize=(6,5))
        ax.text(0.5, 0.5, "No employee data available",
                ha='center', va='center', fontsize=14)
        ax.set_title("Employee Headcount & Turnover", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    df = pd.DataFrame(data)

    # Parse hire_date & termination_date
    df['hire_dt'] = df['hire date'].apply(parse_date)
    df['term_dt'] = df['termination_date'].apply(parse_date)

    # -------------------------------------------------------
    # Left Chart: Headcount Over Time (Monthly Snapshot)
    # -------------------------------------------------------
    date_range = pd.date_range(start="2015-01-01", end="2023-12-01", freq='MS')
    monthly_counts = []
    for snap_date in date_range:
        snap_d = snap_date.date()
        active_count = 0
        for _, row in df.iterrows():
            hire_d = row['hire_dt']
            term_d = row['term_dt']
            if hire_d and hire_d <= snap_d:
                if (term_d is None) or (term_d >= snap_d):
                    active_count += 1
        monthly_counts.append(active_count)

    # -------------------------------------------------------
    # Right Chart: Turnover Rate by Department
    # -------------------------------------------------------
    df['is_terminated'] = df['employment_status'].apply(lambda x: 1 if x == "Terminated" else 0)
    dept_group = df.groupby('department').agg({
        'Employee ID': 'count',
        'is_terminated': 'sum'
    }).rename(columns={'Employee ID': 'dept_count', 'is_terminated': 'dept_term'})
    dept_group['turnover_rate'] = dept_group['dept_term'] / dept_group['dept_count'] * 100
    dept_group_sorted = dept_group.sort_values('turnover_rate', ascending=False)

    # -------------------------------------------------------
    # Make 2-subplot figure with bigger size and spacing
    # -------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    plt.subplots_adjust(wspace=0.3)

    # ------------------ Left: Area Chart -------------------
    # Lighter fill color & alpha
    ax1.fill_between(date_range, monthly_counts, color='steelblue', alpha=0.6)
    ax1.set_title("Headcount Over Time", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Month", fontsize=10)
    ax1.set_ylabel("Active Employees", fontsize=10)

    # Show ticks only once/year
    ax1.xaxis.set_major_locator(mdates.YearLocator())  
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # ----------------- Right: Bar Chart --------------------
    ax2.bar(dept_group_sorted.index,
            dept_group_sorted['turnover_rate'],
            color='orange', alpha=0.8)
    ax2.set_title("Turnover Rate by Department", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Turnover Rate (%)", fontsize=10)
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # ----------------- Final Setup -------------------------
    fig.suptitle("Headcount, Turnover, and Retention", fontsize=14, fontweight='bold')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')


# ---------------------------------------------------------------------
# 2) Compensation: Box Plot of Salary Across Departments
#    /api/employee/compensation
# ---------------------------------------------------------------------
@app.route('/api/employee/compensation', methods=['GET'])
def employee_compensation_chart():
    """
    Box Plot showing salary distribution across departments.
    """
    data = load_employee_data()
    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No employee data available", ha='center', va='center', fontsize=14)
        ax.set_title("Employee Compensation", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    df = pd.DataFrame(data)
    if 'salary' not in df.columns or 'department' not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Missing 'salary' or 'department' field", ha='center', va='center', fontsize=14)
        ax.set_title("Employee Compensation", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    sns.boxplot(x='department', y='salary', data=df, ax=ax)
    ax.set_title("Salary Distribution by Department", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Department", fontsize=11)
    ax.set_ylabel("Salary", fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=9)

    fig.suptitle("Employee Compensation", fontsize=14, fontweight='bold', y=0.98)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')


# ---------------------------------------------------------------------
# 3) Performance: Multi-Gauge for Each Department's Avg Performance vs Target
#    /api/employee/performance
# ---------------------------------------------------------------------
@app.route('/api/employee/performance', methods=['GET'])
def employee_performance_chart():
    """
    Creates a multi-gauge chart showing each department's average performance score vs. a target (8.0).
    We'll display horizontal bars for each department's average score, with a line for the target.
    """
    data = load_employee_data()
    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No employee data available", ha='center', va='center', fontsize=14)
        ax.set_title("Employee Performance", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    df = pd.DataFrame(data)
    if 'department' not in df.columns or 'performance_score' not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Missing 'department' or 'performance_score' field", ha='center', va='center', fontsize=14)
        ax.set_title("Employee Performance", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    # We'll compute average performance_score by department
    dept_perf = df.groupby('department')['performance_score'].mean().sort_values(ascending=False)
    # We'll create horizontal bars. The "target" is 8.0
    target_score = 8.0

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    x_positions = np.arange(len(dept_perf))
    bar_width = 0.6

    bars = ax.barh(
        x_positions,
        dept_perf.values,
        color='green',
        alpha=0.7,
        height=bar_width
    )
    ax.set_yticks(x_positions)
    ax.set_yticklabels(dept_perf.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Avg Performance Score", fontsize=11)
    ax.set_title("Performance vs. Target", fontsize=13, fontweight='bold', pad=10)

    # Draw a vertical line for the target
    ax.axvline(x=target_score, color='red', linestyle='--', label=f"Target = {target_score}")
    ax.legend(loc='lower right', fontsize=9)

    # Annotate each bar with its average
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f"{width:.1f}", va='center', fontsize=9)

    fig.suptitle("Employee Performance (Goal Meters)", fontsize=14, fontweight='bold', y=0.98)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

def load_financial_data():
    with open("financial_data.json", "r") as f:
        data = json.load(f)
    return data

# Optionally parse date if we want time-based grouping
def parse_date(date_str):
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except:
        return None

# -----------------------------------------------------------------------------
# 1) Profit & Loss Endpoint
#    /api/financial/profitloss
# -----------------------------------------------------------------------------
@app.route('/api/financial/profitloss', methods=['GET'])
def financial_profitloss():
    """
    Demonstrates a Profit & Loss style visualization, e.g. a waterfall chart or multi-period line chart,
    grouping ledger categories like 'Sales', 'COGS', 'Expenses', etc.
    For simplicity, we'll do a monthly 'net income' line chart:
      Net Income = sum(positive amounts) + sum(negative amounts)
    """
    data = load_financial_data()
    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No financial data available", ha='center', va='center', fontsize=14)
        ax.set_title("Profit & Loss", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Parse dates
    df['tx_date'] = df['date'].apply(parse_date)

    # We'll group by month-year and sum up amounts => net income for that month
    df['year_month'] = df['tx_date'].apply(lambda d: d.strftime('%Y-%m') if d else None)
    monthly_net = df.groupby('year_month')['amount'].sum().sort_index()
    
    # Plot a line chart for monthly net income
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    # Convert monthly_net to x/y
    x = list(monthly_net.index)
    y = monthly_net.values

    ax.plot(x, y, marker='o', linestyle='-', color='green', alpha=0.8)
    ax.set_title("Monthly Net Income (Profit & Loss)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Net Income", fontsize=11)
    # rotate x
    plt.xticks(rotation=45, ha='right')
    ax.grid(linestyle='--', alpha=0.6)

    fig.suptitle("Profit & Loss Dashboard", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')


# -----------------------------------------------------------------------------
# 2) Balance Sheet & Cash Flow
#    /api/financial/balancesheet
# -----------------------------------------------------------------------------
def load_financial_data():
    """Load JSON data from 'financial_data.json'."""
    with open("financial_data.json", "r") as f:
        return json.load(f)

def parse_date(date_str):
    """Parse a date string (expected format YYYY-MM-DD) into a date object."""
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except Exception as e:
        print(f"[DEBUG] Error parsing date '{date_str}': {e}")
        return None

@app.route('/api/financial/balancesheet', methods=['GET'])
def financial_balancesheet():
    """
    Creates a dual-axis line chart that plots:
      - Assets and Liabilities by month (left y-axis)
      - Net Cash Flow by month (right y-axis)

    The underlying data is computed from financial transactions grouped by month.
    - Assets: Sum of 'amount' for ledger "Assets"
    - Liabilities: Sum of 'amount' for ledger "Liabilities"
    - Net Cash Flow: Sum of 'amount' for ledger "Sales" MINUS Sum for ledger "Expenses"
    """
    data = load_financial_data()
    if not data:
        fig, ax = plt.subplots(figsize=(10,6))
        ax.text(0.5, 0.5, "No financial data available", ha='center', va='center', fontsize=14)
        ax.set_title("Balance Sheet & Cash Flow", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Parse dates from the "date" column
    df['tx_date'] = df['date'].apply(lambda d: parse_date(d))
    # Drop records with unparseable dates
    df = df.dropna(subset=['tx_date'])
    # Create a "year_month" column (e.g., "2021-03")
    df['year_month'] = df['tx_date'].apply(lambda d: d.strftime('%Y-%m'))
    
    # Group by "year_month" and "ledger"
    grouped = df.groupby(['year_month', 'ledger'])['amount'].sum().unstack(fill_value=0)
    
    # For Assets and Liabilities, extract series and reindex by all months
    all_months = sorted(grouped.index)
    assets_series = grouped.get("Assets", pd.Series(0, index=all_months)).reindex(all_months, fill_value=0)
    liab_series = grouped.get("Liabilities", pd.Series(0, index=all_months)).reindex(all_months, fill_value=0)
    
    # Compute Net Cash Flow as: Sales - Expenses
    sales_series = grouped.get("Sales", pd.Series(0, index=all_months)).reindex(all_months, fill_value=0)
    expenses_series = grouped.get("Expenses", pd.Series(0, index=all_months)).reindex(all_months, fill_value=0)
    net_cf_series = sales_series - expenses_series

    # If no data is found after grouping, produce a no-data chart.
    if not all_months:
        fig, ax = plt.subplots(figsize=(10,6))
        ax.text(0.5, 0.5, "No valid monthly data found", ha='center', va='center', fontsize=14)
        ax.set_title("Balance Sheet & Cash Flow", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    
    # Convert the month strings to numeric indices for plotting
    indices = np.arange(len(all_months))
    
    # Create the figure and twin axes
    fig, ax1 = plt.subplots(figsize=(12,6), dpi=120)
    ax2 = ax1.twinx()
    
    # Plot Assets and Liabilities on ax1
    ax1.plot(indices, assets_series.values, marker='o', linestyle='-', color='blue', label='Assets', linewidth=2)
    ax1.plot(indices, liab_series.values, marker='s', linestyle='-', color='red', label='Liabilities', linewidth=2)
    ax1.set_ylabel("Assets / Liabilities", fontsize=11)
    
    # Plot Net Cash Flow on ax2
    ax2.plot(indices, net_cf_series.values, marker='^', linestyle='--', color='green', label='Net CF', linewidth=2, alpha=0.85)
    ax2.set_ylabel("Net Cash Flow", fontsize=11, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Set x-axis labels using the months
    ax1.set_xticks(indices)
    ax1.set_xticklabels(all_months, rotation=45, ha='right', fontsize=9)
    
    # Optionally, if the x-axis is too crowded, show fewer ticks:
    # ax1.xaxis.set_major_locator(plt.MaxNLocator(12))
    
    ax1.set_xlabel("Month", fontsize=11)
    
    # Add grid lines on ax1 for clarity
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    fig.suptitle("Balance Sheet and Cash Flow", fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

# -----------------------------------------------------------------------------
# 3) Budget vs. Actual
#    /api/financial/budget
# -----------------------------------------------------------------------------
@app.route('/api/financial/budget', methods=['GET'])
def financial_budget():
    """
    Demonstrates a bar/column chart comparing actual vs. budget amounts by ledger, or by period.
    We'll do it by ledger category for demonstration.
    """
    data = load_financial_data()
    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No financial data available", ha='center', va='center', fontsize=14)
        ax.set_title("Budget vs. Actual", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    df = pd.DataFrame(data)
    # We'll group by ledger and sum actual vs. budget
    ledger_group = df.groupby('ledger').agg({
        'amount': 'sum',
        'budget_amount': 'sum'
    }).reset_index()

    # For a bar chart with actual vs. budget side by side
    fig, ax = plt.subplots(figsize=(10,6), dpi=120)
    x_vals = np.arange(len(ledger_group))
    bar_width = 0.35

    # Sort by actual descending if desired
    ledger_group = ledger_group.sort_values('amount', ascending=False)

    ax.bar(x_vals - bar_width/2, ledger_group['amount'], bar_width, label='Actual', color='blue', alpha=0.7)
    ax.bar(x_vals + bar_width/2, ledger_group['budget_amount'], bar_width, label='Budget', color='orange', alpha=0.7)

    ax.set_xticks(x_vals)
    ax.set_xticklabels(ledger_group['ledger'], rotation=45, ha='right')
    ax.set_ylabel("Total Amount", fontsize=11)
    ax.set_title("Budget vs. Actual by Ledger", fontsize=13, fontweight='bold', pad=10)
    ax.legend()

    fig.suptitle("Budget vs. Actual", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)
