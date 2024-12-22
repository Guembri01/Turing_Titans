import re
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import List, Any, Dict, Optional


class VisualizationAgent:
    """A class for handling document visualization with multiple plot types."""

    def __init__(self):
        """Initialize the VisualizationAgent."""
        self.data_cache = {}

    def extract_invoice_data(self, document: Any) -> Dict[str, Any]:
        """Extract relevant invoice data from the document using regex."""
        page_content = document.page_content
        metadata = document.metadata

        # Extract Order Date (assuming it's in the format YYYY-MM-DD)
        date_match = re.search(r'Order Date:\s*(\d{4}-\d{2}-\d{2})', page_content)
        date = pd.to_datetime(date_match.group(1)) if date_match else None

        # Extract Total Price
        total_price_match = re.search(r'TotalPrice\s*(\d+\.\d+)', page_content)
        total_price = float(total_price_match.group(1)) if total_price_match else None

        # Extract Order ID (optional, for grouping in bar plots)
        order_id_match = re.search(r'Order ID:\s*(\d+)', page_content)
        order_id = order_id_match.group(1) if order_id_match else None

        return {
            'timestamp': date,
            'value': total_price,
            'order_id': order_id,
            'source': metadata['source']  # Access metadata directly
        }

    def process_time_series_data(self, documents: List[Any]) -> pd.DataFrame:
        """Convert document data to time series format."""
        data = []
        for doc in documents:
            try:
                extracted_data = self.extract_invoice_data(doc)
                if extracted_data['timestamp'] and extracted_data['value']:
                    data.append({
                        'timestamp': extracted_data['timestamp'],
                        'value': extracted_data['value'],
                        'order_id': extracted_data['order_id'],
                        'source': extracted_data['source']
                    })
            except Exception as e:
                st.warning(f"Error processing document: {str(e)}")
                continue
        
        return pd.DataFrame(data).sort_values('timestamp')

    def create_line_plot(self, df: pd.DataFrame, title: str = "Invoice Trends (Line Plot)") -> go.Figure:
        """Create an interactive line plot using Plotly."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='lines+markers',
            name='Invoice Total',
            text=df.apply(lambda row: f"Order ID: {row['order_id']}<br>Source: {row['source']}", axis=1),
            hovertemplate='<b>Date</b>: %{x}<br><b>Total</b>: $%{y:.2f}<br>%{text}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Total Price",
            hovermode='x unified'
        )

        return fig

    def create_bar_plot(self, df: pd.DataFrame, title: str = "Invoice Trends (Bar Plot)") -> go.Figure:
        """Create an interactive bar plot using Plotly, grouped by Order ID."""
        if 'order_id' not in df.columns:
            st.warning("Order ID not found in the data. Bar plot cannot be created.")
            return None

        # Group by Order ID and sum the total price
        grouped_data = df.groupby('order_id')['value'].sum().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=grouped_data['order_id'],
            y=grouped_data['value'],
            text=grouped_data['value'],
            textposition='auto',
            hovertemplate='<b>Order ID</b>: %{x}<br><b>Total</b>: $%{y:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Order ID",
            yaxis_title="Total Price",
            hovermode='x unified'
        )

        return fig

    def create_scatter_plot(self, df: pd.DataFrame, title: str = "Invoice Trends (Scatter Plot)") -> go.Figure:
        """Create an interactive scatter plot using Plotly."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='markers',
            name='Invoice Total',
            text=df.apply(lambda row: f"Order ID: {row['order_id']}<br>Source: {row['source']}", axis=1),
            hovertemplate='<b>Date</b>: %{x}<br><b>Total</b>: $%{y:.2f}<br>%{text}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Total Price",
            hovermode='closest'
        )

        return fig

    def visualize_documents(self, documents: List[Any], plot_type: str = "line", title: Optional[str] = None) -> None:
        """Main method to visualize documents as time series with different plot types."""
        st.write(f"Debug: visualize_documents called with plot_type={plot_type}")  # Debug logging
        df = self.process_time_series_data(documents)
        
        if df.empty:
            st.warning("No valid invoice data found in the documents.")
            return

        # Determine the plot type and create the corresponding figure
        if plot_type == "line":
            fig = self.create_line_plot(df, title or "Invoice Trends (Line Plot)")
        elif plot_type == "bar":
            fig = self.create_bar_plot(df, title or "Invoice Trends (Bar Plot)")
        elif plot_type == "scatter":
            fig = self.create_scatter_plot(df, title or "Invoice Trends (Scatter Plot)")
        else:
            st.error(f"Unsupported plot type: {plot_type}. Supported types are 'line', 'bar', and 'scatter'.")
            return

        if fig:
            st.plotly_chart(fig, use_container_width=True)