"""
PDF report generation for gold price analysis.

This module creates professional PDF reports with charts, forecasts, and market summaries.
"""

import io
import logging
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)


class GoldPriceReportGenerator:
    """Generate PDF reports for gold price analysis."""

    def __init__(self):
        """Initialize report generator."""
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()

    def _create_custom_styles(self) -> dict[str, ParagraphStyle]:
        """Create custom paragraph styles."""
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=self.styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#FFD700"),
            spaceAfter=30,
            alignment=1,  # Center
        )

        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=self.styles["Heading2"],
            fontSize=16,
            textColor=colors.HexColor("#1f77b4"),
            spaceAfter=12,
        )

        return {"title": title_style, "heading": heading_style}

    def generate_monthly_report(
        self,
        current_prices: dict[str, float],
        forecast_data: pd.DataFrame,
        historical_data: pd.DataFrame,
        analysis_metrics: dict[str, Any],
        output_path: str,
    ) -> bool:
        """
        Generate monthly market summary report.

        Args:
            current_prices: Dictionary of current prices by purity
            forecast_data: Forecast DataFrame
            historical_data: Historical price DataFrame
            analysis_metrics: Dictionary with analysis metrics
            output_path: Path to save PDF

        Returns:
            True if successful, False otherwise
        """
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []

            # Title
            title = Paragraph(
                f"Gold Price Analysis Report<br/>{datetime.now().strftime('%B %Y')}",
                self.custom_styles["title"],
            )
            story.append(title)
            story.append(Spacer(1, 0.3 * inch))

            # Executive Summary
            story.append(Paragraph("Executive Summary", self.custom_styles["heading"]))
            story.append(Spacer(1, 0.1 * inch))

            summary_text = f"""
            This report provides a comprehensive analysis of gold prices in Bangladesh for {datetime.now().strftime("%B %Y")}.
            The analysis includes current market prices, historical trends, and 7-day forecasts using advanced machine learning models.
            """
            story.append(Paragraph(summary_text, self.styles["BodyText"]))
            story.append(Spacer(1, 0.2 * inch))

            # Current Prices Table
            story.append(
                Paragraph("Current Market Prices", self.custom_styles["heading"])
            )
            story.append(Spacer(1, 0.1 * inch))

            price_data = [["Purity", "Price (BDT/gram)"]]
            for purity, price in sorted(current_prices.items()):
                price_data.append([purity, f"৳{price:,.2f}"])

            price_table = Table(price_data, colWidths=[2 * inch, 3 * inch])
            price_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 14),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(price_table)
            story.append(Spacer(1, 0.3 * inch))

            # Market Statistics
            if analysis_metrics:
                story.append(
                    Paragraph("Market Statistics", self.custom_styles["heading"])
                )
                story.append(Spacer(1, 0.1 * inch))

                stats_data = [["Metric", "Value"]]
                if "volatility" in analysis_metrics:
                    stats_data.append(
                        ["Volatility (30d)", f"{analysis_metrics['volatility']:.2f}%"]
                    )
                if "trend" in analysis_metrics:
                    stats_data.append(["Trend", analysis_metrics["trend"]])
                if "monthly_change" in analysis_metrics:
                    stats_data.append(
                        ["Monthly Change", f"{analysis_metrics['monthly_change']:.2f}%"]
                    )

                stats_table = Table(stats_data, colWidths=[2.5 * inch, 2.5 * inch])
                stats_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 12),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ]
                    )
                )
                story.append(stats_table)
                story.append(Spacer(1, 0.3 * inch))

            # 7-Day Forecast
            if not forecast_data.empty:
                story.append(
                    Paragraph("7-Day Price Forecast", self.custom_styles["heading"])
                )
                story.append(Spacer(1, 0.1 * inch))

                forecast_table_data = [["Date", "Predicted Price (BDT/gram)"]]
                for _, row in forecast_data.head(7).iterrows():
                    date_str = (
                        row["date"].strftime("%Y-%m-%d")
                        if hasattr(row["date"], "strftime")
                        else str(row["date"])
                    )
                    price_val = row.get("predicted_price", row.get("yhat", 0))
                    forecast_table_data.append([date_str, f"৳{price_val:,.2f}"])

                forecast_table = Table(
                    forecast_table_data, colWidths=[2 * inch, 3 * inch]
                )
                forecast_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ]
                    )
                )
                story.append(forecast_table)
                story.append(Spacer(1, 0.2 * inch))

            # Disclaimer
            story.append(PageBreak())
            story.append(Paragraph("Disclaimer", self.custom_styles["heading"]))
            disclaimer_text = """
            This report is generated for informational purposes only. The forecasts and predictions are based on
            historical data and machine learning models. Actual gold prices may vary significantly from predictions.
            This report should not be used as the sole basis for investment decisions. Please consult with professional
            financial advisors before making any investment decisions.
            """
            story.append(Paragraph(disclaimer_text, self.styles["BodyText"]))

            # Footer
            story.append(Spacer(1, 0.5 * inch))
            footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Gold Price Analyzer"
            story.append(Paragraph(footer_text, self.styles["Italic"]))

            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated successfully: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return False

    def create_chart_image(
        self, fig: go.Figure, width: int = 600, height: int = 400
    ) -> io.BytesIO:
        """
        Convert Plotly figure to image bytes for PDF.

        Args:
            fig: Plotly figure
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            BytesIO object with image data
        """
        try:
            img_bytes = fig.to_image(format="png", width=width, height=height)
            return io.BytesIO(img_bytes)
        except Exception as e:
            logger.error(f"Error creating chart image: {e}")
            return io.BytesIO()

    def add_chart_to_report(
        self, story: list, fig: go.Figure, title: str, width: float = 5 * inch
    ) -> None:
        """
        Add a chart to the PDF report story.

        Args:
            story: Report story list
            fig: Plotly figure
            title: Chart title
            width: Image width
        """
        try:
            story.append(Paragraph(title, self.custom_styles["heading"]))
            story.append(Spacer(1, 0.1 * inch))

            img_buffer = self.create_chart_image(fig)
            if img_buffer.getbuffer().nbytes > 0:
                img = Image(img_buffer, width=width, height=width * 0.6)
                story.append(img)
                story.append(Spacer(1, 0.2 * inch))
        except Exception as e:
            logger.error(f"Error adding chart to report: {e}")


def generate_simple_report(
    current_price: float,
    purity: str,
    forecast_df: pd.DataFrame,
    output_path: str,
) -> bool:
    """
    Generate a simple PDF report (lightweight version).

    Args:
        current_price: Current gold price
        purity: Gold purity
        forecast_df: Forecast DataFrame
        output_path: Output file path

    Returns:
        True if successful
    """
    try:
        generator = GoldPriceReportGenerator()

        current_prices = {purity: current_price}
        analysis_metrics = {}

        return generator.generate_monthly_report(
            current_prices=current_prices,
            forecast_data=forecast_df,
            historical_data=pd.DataFrame(),
            analysis_metrics=analysis_metrics,
            output_path=output_path,
        )
    except Exception as e:
        logger.error(f"Error generating simple report: {e}")
        return False
