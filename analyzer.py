"""
Gold Price Analyzer with Advanced Forecasting
Comprehensive gold price analyzer with 7-day predictive modeling and ML capabilities.
Supports multiple models: Prophet, Random Forest, XGBoost, and ensemble methods.
"""

import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from historical_scraper import HistoricalGoldPriceScraper

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)

# Configure logging
logger = logging.getLogger(__name__)


class AdvancedGoldPriceAnalyzer:
    """
    Comprehensive gold price analyzer with advanced ML forecasting capabilities.
    Supports multiple models: Prophet, Random Forest, XGBoost, and ensemble methods.
    """

    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize the advanced analyzer.

        Args:
            data_dir: Directory containing price data
            models_dir: Directory to store trained models
        """
        # Data directory setup
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.scaler = MinMaxScaler()

        # Models directory
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # Historical data scraper
        self.historical_scraper = HistoricalGoldPriceScraper(data_dir)

        # Model storage
        self.trained_models = {}
        self.model_metadata = {}

    def load_data(self, source: str = "duckdb", days: int = 365) -> pd.DataFrame:
        """
        Load historical price data from storage.

        Args:
            source: Data source ('duckdb' or 'csv')
            days: Number of days to load

        Returns:
            DataFrame with historical data
        """
        try:
            if source == "duckdb":
                db_path = self.data_dir / "gold_prices.db"
                if db_path.exists():
                    conn = duckdb.connect(str(db_path))
                    query = f"""
                        SELECT * FROM prices
                        WHERE CAST(date AS DATE) >= current_date - INTERVAL '{days} days'
                        ORDER BY timestamp ASC
                    """
                    df = conn.execute(query).df()
                    conn.close()
                    if not df.empty:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df["date"] = pd.to_datetime(df["date"])
                        return df

            elif source == "csv":
                csv_files = list(self.data_dir.glob("gold_prices_*.csv"))
                if csv_files:
                    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                    df = pd.read_csv(latest_csv)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df["date"] = pd.to_datetime(df["date"])
                    return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")

        return pd.DataFrame()

    def calculate_statistics(
        self, df: pd.DataFrame, metal: str = "gold", purity: str = "22K"
    ) -> dict[str, float]:
        """
        Calculate basic statistics for a specific metal and purity.

        Args:
            df: DataFrame with price data
            metal: Metal type
            purity: Purity level

        Returns:
            Dictionary with calculated statistics
        """
        if df.empty:
            return {}

        filtered_df = df[(df["metal"] == metal) & (df["purity"] == purity)]

        if filtered_df.empty:
            return {}

        prices = filtered_df["price_bdt_per_gram"]

        stats = {
            "current_price": float(prices.iloc[-1]) if not prices.empty else 0.0,
            "average_price": float(prices.mean()),
            "min_price": float(prices.min()),
            "max_price": float(prices.max()),
            "price_volatility": float(prices.std()),
            "total_records": len(prices),
            "latest_date": (
                filtered_df["date"].max().strftime("%Y-%m-%d")
                if not filtered_df.empty
                else ""
            ),
        }

        # Calculate price changes
        if len(prices) > 1:
            stats["daily_change"] = float(prices.iloc[-1] - prices.iloc[-2])
            stats["daily_change_percent"] = float(
                (stats["daily_change"] / prices.iloc[-2]) * 100
            )
        else:
            stats["daily_change"] = 0.0
            stats["daily_change_percent"] = 0.0

        # Calculate weekly change
        if len(prices) > 7:
            week_ago_price = prices.iloc[-8]
            stats["weekly_change"] = float(prices.iloc[-1] - week_ago_price)
            stats["weekly_change_percent"] = float(
                (stats["weekly_change"] / week_ago_price) * 100
            )
        else:
            stats["weekly_change"] = 0.0
            stats["weekly_change_percent"] = 0.0

        return stats

    def calculate_trends(
        self,
        df: pd.DataFrame,
        metal: str = "gold",
        purity: str = "22K",
        window: int = 7,
    ) -> pd.DataFrame:
        """
        Calculate moving averages and trend indicators.

        Args:
            df: DataFrame with price data
            metal: Metal type
            purity: Purity level
            window: Moving average window size

        Returns:
            DataFrame with trend calculations
        """
        if df.empty:
            return pd.DataFrame()

        filtered_df = df[(df["metal"] == metal) & (df["purity"] == purity)].copy()

        if filtered_df.empty:
            return pd.DataFrame()

        # Sort by date
        filtered_df = filtered_df.sort_values("date")

        # Calculate moving averages
        filtered_df[f"ma_{window}"] = (
            filtered_df["price_bdt_per_gram"].rolling(window=window).mean()
        )
        filtered_df[f"ma_{window * 2}"] = (
            filtered_df["price_bdt_per_gram"].rolling(window=window * 2).mean()
        )

        # Calculate price changes
        filtered_df["price_change"] = filtered_df["price_bdt_per_gram"].diff()
        filtered_df["price_change_percent"] = (
            filtered_df["price_change"] / filtered_df["price_bdt_per_gram"].shift(1)
        ) * 100

        # Trend direction
        filtered_df["trend_direction"] = np.where(
            filtered_df[f"ma_{window}"] > filtered_df[f"ma_{window * 2}"],
            "Upward",
            "Downward",
        )

        return filtered_df

    def create_price_chart(
        self,
        df: pd.DataFrame,
        metal: str = "gold",
        purity: str = "22K",
        forecast: pd.DataFrame | None = None,
    ) -> go.Figure:
        """
        Create an interactive price chart using Plotly.

        Args:
            df: Historical price data
            metal: Metal type
            purity: Purity level
            forecast: Optional forecast data to overlay

        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                f"{metal.title()} Price Trend ({purity})",
                "Price Change %",
            ],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
        )

        if not df.empty:
            filtered_df = df[(df["metal"] == metal) & (df["purity"] == purity)]

            if not filtered_df.empty:
                # Historical prices
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df["date"],
                        y=filtered_df["price_bdt_per_gram"],
                        mode="lines+markers",
                        name="Historical Price",
                        line=dict(color="#1f77b4", width=2),
                        marker=dict(size=4),
                    ),
                    row=1,
                    col=1,
                )

                # Moving average
                if len(filtered_df) > 7:
                    ma_7 = filtered_df["price_bdt_per_gram"].rolling(window=7).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df["date"],
                            y=ma_7,
                            mode="lines",
                            name="7-day MA",
                            line=dict(color="#ff7f0e", width=1, dash="dash"),
                        ),
                        row=1,
                        col=1,
                    )

                # Price change percentage
                price_change_pct = filtered_df["price_bdt_per_gram"].pct_change() * 100
                colors = ["red" if x < 0 else "green" for x in price_change_pct]

                fig.add_trace(
                    go.Bar(
                        x=filtered_df["date"],
                        y=price_change_pct,
                        name="Daily Change %",
                        marker_color=colors,
                        opacity=0.7,
                    ),
                    row=2,
                    col=1,
                )

        # Add forecast if provided
        if forecast is not None and not forecast.empty:
            future_forecast = (
                forecast[forecast["ds"] > df["date"].max()]
                if not df.empty
                else forecast
            )

            if not future_forecast.empty:
                fig.add_trace(
                    go.Scatter(
                        x=future_forecast["ds"],
                        y=future_forecast["yhat"],
                        mode="lines",
                        name="Forecast",
                        line=dict(color="#d62728", width=2, dash="dot"),
                    ),
                    row=1,
                    col=1,
                )

                # Confidence intervals
                fig.add_trace(
                    go.Scatter(
                        x=future_forecast["ds"],
                        y=future_forecast["yhat_upper"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=future_forecast["ds"],
                        y=future_forecast["yhat_lower"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(214, 39, 40, 0.2)",
                        name="Confidence Interval",
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )

        # Update layout
        fig.update_layout(
            title=f"{metal.title()} Price Analysis ({purity})",
            xaxis_title="Date",
            height=600,
            showlegend=True,
        )

        fig.update_yaxes(title="Price (BDT/gram)", row=1, col=1)
        fig.update_yaxes(title="Change %", row=2, col=1)

        return fig

    def preprocess_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess historical data for ML training.

        Args:
            df: Raw historical data DataFrame

        Returns:
            Preprocessed DataFrame with features
        """
        if df.empty:
            return df

        logger.info("Preprocessing historical data...")

        # Ensure proper data types
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["price_bdt_per_gram"] = pd.to_numeric(
            df["price_bdt_per_gram"], errors="coerce"
        )

        # Remove invalid data
        df = df.dropna(subset=["price_bdt_per_gram"])
        df = df[df["price_bdt_per_gram"] > 0]

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        # Handle missing dates (interpolation)
        df = self._fill_missing_dates(df)

        # Calculate derived features
        df = self._calculate_technical_features(df)

        logger.info(
            f"Preprocessed {len(df)} records with {df.columns.tolist()} features"
        )
        return df

    def _fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing dates using interpolation."""
        if df.empty:
            return df

        # Create complete date range
        date_range = pd.date_range(
            start=df["date"].min(), end=df["date"].max(), freq="D"
        )

        # Create complete DataFrame
        complete_df = pd.DataFrame({"date": date_range})

        # Merge with existing data
        merged_df = complete_df.merge(df, on="date", how="left")

        # Interpolate missing prices
        merged_df["price_bdt_per_gram"] = merged_df["price_bdt_per_gram"].interpolate(
            method="linear", limit_direction="both"
        )

        # Fill other columns
        for col in ["purity", "metal", "source"]:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].ffill().bfill()

        return merged_df.dropna(subset=["price_bdt_per_gram"])

    def _calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical analysis features."""
        if df.empty or "price_bdt_per_gram" not in df.columns:
            return df

        df = df.copy()

        # Moving averages
        df["ma_7"] = df["price_bdt_per_gram"].rolling(window=7, min_periods=1).mean()
        df["ma_14"] = df["price_bdt_per_gram"].rolling(window=14, min_periods=1).mean()
        df["ma_30"] = df["price_bdt_per_gram"].rolling(window=30, min_periods=1).mean()

        # Price changes
        df["daily_change"] = df["price_bdt_per_gram"].diff()
        df["daily_change_pct"] = df["price_bdt_per_gram"].pct_change() * 100

        # Volatility (rolling standard deviation)
        df["volatility_7"] = (
            df["price_bdt_per_gram"].rolling(window=7, min_periods=1).std()
        )
        df["volatility_14"] = (
            df["price_bdt_per_gram"].rolling(window=14, min_periods=1).std()
        )

        # Relative Strength Index (RSI) approximation
        delta = df["daily_change"].fillna(0)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-8)  # Avoid division by zero
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["bb_upper"] = df["ma_14"] + (2 * df["volatility_14"])
        df["bb_lower"] = df["ma_14"] - (2 * df["volatility_14"])
        df["bb_position"] = (df["price_bdt_per_gram"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        # Lag features (previous days' prices)
        for lag in [1, 2, 3, 7]:
            df[f"price_lag_{lag}"] = df["price_bdt_per_gram"].shift(lag)

        # Time-based features
        df["day_of_week"] = df["date"].dt.dayofweek  # type: ignore[attr-defined]
        df["month"] = df["date"].dt.month  # type: ignore[attr-defined]
        df["quarter"] = df["date"].dt.quarter  # type: ignore[attr-defined]

        # Fill NaN values
        df = df.bfill().ffill()

        return df

    def detect_ma_crossovers(
        self, df: pd.DataFrame, short_window: int = 7, long_window: int = 30
    ) -> pd.DataFrame:
        """
        Detect Golden Cross and Death Cross signals.

        Golden Cross: Short MA crosses above Long MA (Buy signal)
        Death Cross: Short MA crosses below Long MA (Sell signal)

        Args:
            df: DataFrame with price data and moving averages
            short_window: Short-term MA window (default 7)
            long_window: Long-term MA window (default 30)

        Returns:
            DataFrame with crossover signals
        """
        if (
            df.empty
            or f"ma_{short_window}" not in df.columns
            or f"ma_{long_window}" not in df.columns
        ):
            logger.warning("Missing MA columns for crossover detection")
            return df

        df = df.copy()

        # Calculate crossover signals
        df["ma_short"] = df[f"ma_{short_window}"]
        df["ma_long"] = df[f"ma_{long_window}"]

        # Detect when short MA is above/below long MA
        df["short_above_long"] = df["ma_short"] > df["ma_long"]

        # Detect crossovers by comparing current and previous state
        df["crossover"] = df["short_above_long"].astype(int).diff()

        # Create signal column
        df["signal"] = "Hold"
        df.loc[df["crossover"] == 1, "signal"] = "Golden Cross (Buy)"
        df.loc[df["crossover"] == -1, "signal"] = "Death Cross (Sell)"

        # Calculate signal strength based on price momentum
        if "price_change_pct" in df.columns:
            df["signal_strength"] = abs(df["price_change_pct"])
        else:
            df["signal_strength"] = 0.0

        # Clean up temporary columns
        df = df.drop(columns=["short_above_long", "crossover"], errors="ignore")

        logger.info(
            f"Detected {len(df[df['signal'] == 'Golden Cross (Buy)'])} Golden Cross and "
            f"{len(df[df['signal'] == 'Death Cross (Sell)'])} Death Cross signals"
        )

        return df

    def get_latest_signal(self, df: pd.DataFrame) -> dict[str, any]:
        """
        Get the most recent trading signal.

        Args:
            df: DataFrame with crossover signals

        Returns:
            Dictionary with latest signal information
        """
        if df.empty or "signal" not in df.columns:
            return {"signal": "No Data", "date": None, "strength": 0.0}

        # Get last non-Hold signal
        signal_df = df[df["signal"] != "Hold"].copy()

        if signal_df.empty:
            return {
                "signal": "Hold",
                "date": df.iloc[-1]["date"] if "date" in df.columns else None,
                "strength": 0.0,
                "ma_short": df.iloc[-1].get("ma_short", 0),
                "ma_long": df.iloc[-1].get("ma_long", 0),
            }

        latest = signal_df.iloc[-1]

        return {
            "signal": latest["signal"],
            "date": latest.get("date", None),
            "strength": latest.get("signal_strength", 0.0),
            "ma_short": latest.get("ma_short", 0),
            "ma_long": latest.get("ma_long", 0),
            "price": latest.get("price_bdt_per_gram", 0),
        }

    def prepare_ml_dataset(
        self, df: pd.DataFrame, forecast_days: int = 7
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Prepare dataset for machine learning training.

        Args:
            df: Preprocessed historical data
            forecast_days: Number of days to forecast

        Returns:
            Tuple of (features, targets, feature_names)
        """
        if df.empty:
            return np.array([]), np.array([]), []

        # Feature columns (exclude target and non-feature columns)
        exclude_cols = ["date", "price_bdt_per_gram", "purity", "metal", "source"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Prepare features and targets
        features = []
        targets = []

        for i in range(len(df) - forecast_days):
            # Features: current day data
            feature_row = df.iloc[i][feature_cols].values

            # Target: average price over next forecast_days
            future_prices = df.iloc[i + 1 : i + 1 + forecast_days][
                "price_bdt_per_gram"
            ].values
            target = (
                np.mean(future_prices)  # type: ignore[arg-type]
                if len(future_prices) > 0
                else df.iloc[i]["price_bdt_per_gram"]
            )

            features.append(feature_row)
            targets.append(target)

        X = np.array(features)
        y = np.array(targets)

        # Handle NaN values
        if X.size > 0:
            X = np.nan_to_num(X, nan=0.0)
        if y.size > 0:
            y = np.nan_to_num(y, nan=0.0)

        logger.info(f"Prepared ML dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_cols

    def train_prophet_model(
        self, df: pd.DataFrame, purity: str = "22K"
    ) -> Prophet | None:
        """
        Train Prophet model for time series forecasting.

        Args:
            df: Historical data DataFrame
            purity: Gold purity to train on

        Returns:
            Trained Prophet model
        """
        if df.empty:
            return None

        logger.info(f"Training Prophet model for {purity} gold...")

        try:
            # Filter by purity
            purity_df = (
                df[df["purity"] == purity].copy()
                if "purity" in df.columns
                else df.copy()
            )

            if len(purity_df) < 10:
                logger.warning(
                    f"Insufficient data for Prophet training ({len(purity_df)} records)"
                )
                return None

            # Prepare Prophet data format
            prophet_df = pd.DataFrame(
                {
                    "ds": pd.to_datetime(purity_df["date"]),
                    "y": purity_df["price_bdt_per_gram"],
                }
            )

            # Configure Prophet model
            model = Prophet(
                yearly_seasonality=True,  # type: ignore[arg-type]
                weekly_seasonality=True,  # type: ignore[arg-type]
                daily_seasonality=False,  # type: ignore[arg-type]
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=0.95,
                seasonality_mode="multiplicative",
            )

            # Add custom seasonalities
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

            # Train model
            model.fit(prophet_df)

            # Save model
            model_path = self.models_dir / f"prophet_model_{purity.lower()}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            self.trained_models[f"prophet_{purity}"] = model
            self.model_metadata[f"prophet_{purity}"] = {
                "type": "prophet",
                "purity": purity,
                "training_samples": len(prophet_df),
                "trained_at": datetime.now().isoformat(),
                "model_path": str(model_path),
            }

            logger.info(f"Prophet model trained and saved for {purity}")
            return model

        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return None

    def train_random_forest_model(
        self, df: pd.DataFrame, purity: str = "22K"
    ) -> RandomForestRegressor | None:
        """
        Train Random Forest model for price prediction.

        Args:
            df: Preprocessed historical data
            purity: Gold purity to train on

        Returns:
            Trained Random Forest model
        """
        if df.empty:
            return None

        logger.info(f"Training Random Forest model for {purity} gold...")

        try:
            # Filter by purity
            purity_df = (
                df[df["purity"] == purity].copy()
                if "purity" in df.columns
                else df.copy()
            )

            if len(purity_df) < 20:
                logger.warning(
                    f"Insufficient data for Random Forest training ({len(purity_df)} records)"
                )
                return None

            # Prepare ML dataset
            X, y, feature_names = self.prepare_ml_dataset(purity_df, forecast_days=7)

            if X.size == 0:
                logger.warning("No valid features for Random Forest training")
                return None

            # Train Random Forest
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X, y)

            # Save model and scaler
            model_path = self.models_dir / f"rf_model_{purity.lower()}.pkl"
            joblib.dump(model, model_path)

            # Save feature names
            feature_path = self.models_dir / f"rf_features_{purity.lower()}.pkl"
            joblib.dump(feature_names, feature_path)

            self.trained_models[f"rf_{purity}"] = model
            self.model_metadata[f"rf_{purity}"] = {
                "type": "random_forest",
                "purity": purity,
                "training_samples": len(X),
                "features": feature_names,
                "trained_at": datetime.now().isoformat(),
                "model_path": str(model_path),
                "feature_path": str(feature_path),
            }

            logger.info(f"Random Forest model trained and saved for {purity}")
            return model

        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            return None

    def generate_7_day_forecast(
        self, purity: str = "22K", model_type: str = "ensemble"
    ) -> dict:
        """
        Generate 7-day price forecast using trained models.

        Args:
            purity: Gold purity to forecast
            model_type: Type of model ('prophet', 'random_forest', 'ensemble')

        Returns:
            Dictionary with forecast results
        """
        logger.info(
            f"Generating 7-day forecast for {purity} gold using {model_type} model"
        )

        try:
            # Load historical data
            historical_df = self.historical_scraper.load_historical_data(days=365)

            if historical_df.empty:
                logger.warning("No historical data available for forecasting")
                return {"error": "No historical data available"}

            # Preprocess data
            processed_df = self.preprocess_historical_data(historical_df)

            if processed_df.empty:
                return {"error": "No valid data after preprocessing"}

            # Generate forecasts based on model type
            forecast_results = {
                "purity": purity,
                "model_type": model_type,
                "forecast_date": datetime.now().isoformat(),
                "forecasts": [],
                "confidence_intervals": [],
                "metadata": {},
            }

            if model_type in ["prophet", "ensemble"]:
                prophet_forecast = self._generate_prophet_forecast(processed_df, purity)
                if prophet_forecast:
                    forecast_results["prophet"] = prophet_forecast

            if model_type in ["random_forest", "ensemble"]:
                rf_forecast = self._generate_rf_forecast(processed_df, purity)
                if rf_forecast:
                    forecast_results["random_forest"] = rf_forecast

            # Create ensemble forecast if both models available
            if (
                model_type == "ensemble"
                and "prophet" in forecast_results
                and "random_forest" in forecast_results
            ):
                ensemble_forecast = self._create_ensemble_forecast(
                    forecast_results["prophet"], forecast_results["random_forest"]
                )
                forecast_results["ensemble"] = ensemble_forecast
                forecast_results["forecasts"] = ensemble_forecast["forecasts"]
            elif "prophet" in forecast_results:
                forecast_results["forecasts"] = forecast_results["prophet"]["forecasts"]
            elif "random_forest" in forecast_results:
                forecast_results["forecasts"] = forecast_results["random_forest"][
                    "forecasts"
                ]

            return forecast_results

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {"error": str(e)}

    def _generate_prophet_forecast(self, df: pd.DataFrame, purity: str) -> dict | None:
        """Generate forecast using Prophet model."""
        try:
            # Load or train Prophet model
            model_key = f"prophet_{purity}"
            model = self.trained_models.get(model_key)

            if model is None:
                model = self.train_prophet_model(df, purity)
                if model is None:
                    return None

            # Create future dataframe for 7 days
            future = model.make_future_dataframe(periods=7)
            forecast = model.predict(future)

            # Extract future predictions
            future_forecast = forecast.tail(7)

            forecasts = []
            for _, row in future_forecast.iterrows():
                forecasts.append(
                    {
                        "date": row["ds"].date().isoformat(),
                        "predicted_price": float(row["yhat"]),
                        "lower_bound": float(row["yhat_lower"]),
                        "upper_bound": float(row["yhat_upper"]),
                        "confidence": 0.95,
                    }
                )

            return {
                "model_type": "prophet",
                "forecasts": forecasts,
                "trend": (
                    "upward"
                    if forecast.iloc[-1]["trend"] > forecast.iloc[-8]["trend"]
                    else "downward"
                ),
            }

        except Exception as e:
            logger.error(f"Error in Prophet forecast: {e}")
            return None

    def _generate_rf_forecast(self, df: pd.DataFrame, purity: str) -> dict | None:
        """Generate forecast using Random Forest model."""
        try:
            # Load or train Random Forest model
            model_key = f"rf_{purity}"
            model = self.trained_models.get(model_key)

            if model is None:
                model = self.train_random_forest_model(df, purity)
                if model is None:
                    return None

            # Get latest data for prediction
            purity_df = (
                df[df["purity"] == purity].copy()
                if "purity" in df.columns
                else df.copy()
            )
            latest_data = purity_df.tail(1)

            if latest_data.empty:
                return None

            # Load feature names
            feature_path = self.models_dir / f"rf_features_{purity.lower()}.pkl"
            if feature_path.exists():
                feature_names = joblib.load(feature_path)
            else:
                return None

            # Generate 7-day forecast
            forecasts = []
            current_price = latest_data["price_bdt_per_gram"].iloc[0]

            for day in range(1, 8):
                # Prepare features for prediction
                feature_values = latest_data[feature_names].values.reshape(1, -1)

                # Make prediction
                predicted_price = model.predict(feature_values)[0]

                # Add some uncertainty (Random Forest doesn't provide native uncertainty)
                std_dev = current_price * 0.02  # 2% standard deviation
                lower_bound = predicted_price - (1.96 * std_dev)
                upper_bound = predicted_price + (1.96 * std_dev)

                forecast_date = (datetime.now() + timedelta(days=day)).date()

                forecasts.append(
                    {
                        "date": forecast_date.isoformat(),
                        "predicted_price": float(predicted_price),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "confidence": 0.95,
                    }
                )

                # Update current price for next iteration
                current_price = predicted_price

            return {
                "model_type": "random_forest",
                "forecasts": forecasts,
                "feature_importance": (
                    dict(zip(feature_names, model.feature_importances_))
                    if hasattr(model, "feature_importances_")
                    else {}
                ),
            }

        except Exception as e:
            logger.error(f"Error in Random Forest forecast: {e}")
            return None

    def _create_ensemble_forecast(
        self, prophet_forecast: dict, rf_forecast: dict
    ) -> dict:
        """Create ensemble forecast by combining Prophet and Random Forest predictions."""
        ensemble_forecasts = []

        for i in range(
            min(len(prophet_forecast["forecasts"]), len(rf_forecast["forecasts"]))
        ):
            prophet_pred = prophet_forecast["forecasts"][i]
            rf_pred = rf_forecast["forecasts"][i]

            # Weighted average (Prophet: 60%, Random Forest: 40%)
            ensemble_price = (0.6 * prophet_pred["predicted_price"]) + (
                0.4 * rf_pred["predicted_price"]
            )

            # Combined confidence intervals
            lower_bound = min(prophet_pred["lower_bound"], rf_pred["lower_bound"])
            upper_bound = max(prophet_pred["upper_bound"], rf_pred["upper_bound"])

            ensemble_forecasts.append(
                {
                    "date": prophet_pred["date"],
                    "predicted_price": float(ensemble_price),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "confidence": 0.90,  # Slightly lower confidence for ensemble
                    "prophet_prediction": prophet_pred["predicted_price"],
                    "rf_prediction": rf_pred["predicted_price"],
                }
            )

        return {
            "model_type": "ensemble",
            "forecasts": ensemble_forecasts,
            "weights": {"prophet": 0.6, "random_forest": 0.4},
        }

    def evaluate_model_accuracy(self, purity: str = "22K", test_days: int = 30) -> dict:
        """
        Evaluate model accuracy using historical data.

        Args:
            purity: Gold purity to evaluate
            test_days: Number of days to use for testing

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model accuracy for {purity} gold")

        try:
            # Load historical data
            historical_df = self.historical_scraper.load_historical_data(days=365)

            if len(historical_df) < test_days + 30:
                return {"error": "Insufficient data for evaluation"}

            # Preprocess data
            processed_df = self.preprocess_historical_data(historical_df)

            # Split data
            train_df = processed_df.iloc[:-test_days]
            test_df = processed_df.iloc[-test_days:]

            # Train models on training data
            prophet_model = self.train_prophet_model(train_df, purity)
            rf_model = self.train_random_forest_model(train_df, purity)

            evaluation_results = {
                "purity": purity,
                "test_period": test_days,
                "evaluation_date": datetime.now().isoformat(),
                "models": {},
            }

            # Evaluate Prophet model
            if prophet_model:
                prophet_metrics = self._evaluate_prophet_model(
                    prophet_model, test_df, purity
                )
                evaluation_results["models"]["prophet"] = prophet_metrics

            # Evaluate Random Forest model
            if rf_model:
                rf_metrics = self._evaluate_rf_model(rf_model, test_df, purity)
                evaluation_results["models"]["random_forest"] = rf_metrics

            return evaluation_results

        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            return {"error": str(e)}

    def _evaluate_prophet_model(
        self, model: Prophet, test_df: pd.DataFrame, purity: str
    ) -> dict:
        """Evaluate Prophet model performance."""
        try:
            purity_df = (
                test_df[test_df["purity"] == purity].copy()
                if "purity" in test_df.columns
                else test_df.copy()
            )

            if purity_df.empty:
                return {"error": "No test data for specified purity"}

            # Prepare test data
            test_prophet_df = pd.DataFrame(
                {
                    "ds": pd.to_datetime(purity_df["date"]),
                    "y": purity_df["price_bdt_per_gram"],
                }
            )

            # Generate predictions
            forecast = model.predict(test_prophet_df)

            # Calculate metrics
            y_true = test_prophet_df["y"].values
            y_pred = forecast["yhat"].values

            mae = mean_absolute_error(y_true, y_pred)  # type: ignore[arg-type]
            mse = mean_squared_error(y_true, y_pred)  # type: ignore[arg-type]
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # type: ignore[operator]
            r2 = r2_score(y_true, y_pred)  # type: ignore[arg-type]

            return {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "mape": float(mape),
                "r2_score": float(r2),
                "test_samples": len(y_true),
                "accuracy_percentage": float(100 - mape),
            }

        except Exception as e:
            logger.error(f"Error evaluating Prophet model: {e}")
            return {"error": str(e)}

    def _evaluate_rf_model(
        self, model: RandomForestRegressor, test_df: pd.DataFrame, purity: str
    ) -> dict:
        """Evaluate Random Forest model performance."""
        try:
            purity_df = (
                test_df[test_df["purity"] == purity].copy()
                if "purity" in test_df.columns
                else test_df.copy()
            )

            if purity_df.empty:
                return {"error": "No test data for specified purity"}

            # Load feature names
            feature_path = self.models_dir / f"rf_features_{purity.lower()}.pkl"
            if not feature_path.exists():
                return {"error": "Feature names not found"}

            feature_names = joblib.load(feature_path)

            # Prepare test features
            X_test = purity_df[feature_names].values
            y_test = purity_df["price_bdt_per_gram"].values

            # Generate predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)  # type: ignore[arg-type]
            mse = mean_squared_error(y_test, y_pred)  # type: ignore[arg-type]
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            r2 = r2_score(y_test, y_pred)  # type: ignore[arg-type]

            return {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "mape": float(mape),
                "r2_score": float(r2),
                "test_samples": len(y_test),
                "accuracy_percentage": float(100 - mape),
                "feature_importance": dict(
                    zip(feature_names, model.feature_importances_)
                ),
            }

        except Exception as e:
            logger.error(f"Error evaluating Random Forest model: {e}")
            return {"error": str(e)}

    def create_forecast_visualization(
        self, forecast_results: dict, historical_df: pd.DataFrame | None = None
    ) -> go.Figure:
        """
        Create interactive visualization for forecast results.

        Args:
            forecast_results: Forecast results from generate_7_day_forecast
            historical_df: Optional historical data for context

        Returns:
            Plotly figure with forecast visualization
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["7-Day Gold Price Forecast", "Daily Price Changes"],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3],
        )

        # Historical data
        if historical_df is not None and not historical_df.empty:
            recent_data = historical_df.tail(30)  # Last 30 days

            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(recent_data["date"]),
                    y=recent_data["price_bdt_per_gram"],
                    mode="lines+markers",
                    name="Historical Prices",
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=4),
                ),
                row=1,
                col=1,
            )

        # Forecast data
        if "forecasts" in forecast_results and forecast_results["forecasts"]:
            forecasts = forecast_results["forecasts"]

            forecast_dates = [pd.to_datetime(f["date"]) for f in forecasts]
            forecast_prices = [f["predicted_price"] for f in forecasts]
            lower_bounds = [f["lower_bound"] for f in forecasts]
            upper_bounds = [f["upper_bound"] for f in forecasts]

            # Forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast_prices,
                    mode="lines+markers",
                    name="7-Day Forecast",
                    line=dict(color="#d62728", width=3, dash="dot"),
                    marker=dict(size=6, symbol="diamond"),
                ),
                row=1,
                col=1,
            )

            # Confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates + forecast_dates[::-1],
                    y=upper_bounds + lower_bounds[::-1],
                    fill="toself",
                    fillcolor="rgba(214, 39, 40, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Confidence Interval",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Daily changes
            daily_changes = []
            for i in range(1, len(forecast_prices)):
                change = forecast_prices[i] - forecast_prices[i - 1]
                daily_changes.append(change)

            if daily_changes:
                colors = ["green" if x > 0 else "red" for x in daily_changes]

                fig.add_trace(
                    go.Bar(
                        x=forecast_dates[1:],
                        y=daily_changes,
                        name="Predicted Daily Changes",
                        marker_color=colors,
                        opacity=0.7,
                    ),
                    row=2,
                    col=1,
                )

        # Update layout
        fig.update_layout(
            title=f"Gold Price 7-Day Forecast ({forecast_results.get('purity', '22K')})",
            height=600,
            showlegend=True,
        )

        fig.update_yaxes(title="Price (BDT/gram)", row=1, col=1)
        fig.update_yaxes(title="Daily Change (BDT)", row=2, col=1)
        fig.update_xaxes(title="Date", row=2, col=1)

        return fig


async def main():
    """Example usage of the AdvancedGoldPriceAnalyzer."""
    analyzer = AdvancedGoldPriceAnalyzer()

    # Fetch and preprocess historical data
    print("Fetching historical data...")
    historical_df = await analyzer.historical_scraper.fetch_all_historical_data(
        days=180
    )

    if not historical_df.empty:
        # Save historical data
        analyzer.historical_scraper.save_historical_data(historical_df)
        print(f"Saved {len(historical_df)} historical records")

        # Generate 7-day forecast
        print("\\nGenerating 7-day forecast...")
        forecast_results = analyzer.generate_7_day_forecast(
            purity="22K", model_type="ensemble"
        )

        if "error" not in forecast_results:
            print("7-Day Forecast Results:")
            for forecast in forecast_results["forecasts"][:3]:  # Show first 3 days
                print(
                    f"  {forecast['date']}: ৳{forecast['predicted_price']:,.0f} "
                    f"({forecast['lower_bound']:,.0f} - {forecast['upper_bound']:,.0f})"
                )

        # Evaluate model accuracy
        print("\\nEvaluating model accuracy...")
        evaluation = analyzer.evaluate_model_accuracy(purity="22K", test_days=14)

        if "error" not in evaluation and "models" in evaluation:
            for model_name, metrics in evaluation["models"].items():
                if "error" not in metrics:
                    print(
                        f"  {model_name}: {metrics.get('accuracy_percentage', 0):.1f}% accuracy"
                    )
    else:
        print("No historical data available")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
