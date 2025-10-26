import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from utilsforecast.plotting import plot_series
from utilsforecast.losses import mae, mape, rmse, smape
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from mlforecast.utils import PredictionIntervals
from utilsforecast.evaluation import evaluate
import io

# Set random seeds for reproducibility
np.random.seed(42)

def evaluate_crossvalidation(crossvalidation_df, metrics, models):
    evaluations = []
    for c in crossvalidation_df['cutoff'].unique():
        df_cv = crossvalidation_df.query('cutoff == @c')
        evaluation = evaluate(
            df=df_cv,
            metrics=metrics,
            models=list(models.keys())
        )
        evaluations.append(evaluation)
    evaluations = pd.concat(evaluations, ignore_index=True).drop(columns='unique_id')
    evaluations = evaluations.groupby('metric').mean()
    return evaluations

def calculate_ape(actual, predicted):
    """Calculate Absolute Percentage Error"""
    return np.abs((actual - predicted) / actual) * 100

def create_forecast_vs_actual_table(df_actual, df_forecast, models):
    """Create detailed forecast vs actual comparison table"""
    # Merge actual and forecast data
    comparison = df_actual[['ds', 'y']].merge(
        df_forecast[['ds'] + list(models.keys())], 
        on='ds', 
        how='inner'
    )
    
    # Calculate residuals and APE for each model
    for model in models.keys():
        comparison[f'{model}_residual'] = comparison['y'] - comparison[model]
        comparison[f'{model}_APE'] = calculate_ape(comparison['y'], comparison[model])
    
    return comparison

# Set page config with custom theme
st.set_page_config(
    page_title="AISPIRE - Advanced Time Series Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2E86AB;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #4CAF50;
        margin-bottom: 2rem;
    }
    h2 {
        color: #2E86AB;
        border-left: 5px solid #4CAF50;
        padding-left: 15px;
        margin-top: 2rem;
    }
    h3 {
        color: #555;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with icon
st.markdown("<h1>🚀 AISPIRE - Advanced Time Series Forecasting Platform</h1>", unsafe_allow_html=True)

# Initialize session state
if 'forecast_generated' not in st.session_state:
    st.session_state.forecast_generated = False
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = None
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'comparison_table' not in st.session_state:
    st.session_state.comparison_table = None

# Cache raw data loading
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def process_data(df, date_column, value_column):
    df = df[[date_column, value_column]].copy()
    df.columns = ['ds', 'y']
    df.insert(0, 'unique_id', 'series')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Convert target column to numeric
    original_rows = len(df)
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna()
    rows_lost = original_rows - len(df)
    
    if rows_lost > 0:
        st.warning(f"⚠️ {rows_lost} rows removed due to missing/invalid values ({rows_lost/original_rows*100:.1f}%)")
    
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    return df

# Sidebar configuration
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/4CAF50/FFFFFF?text=AISPIRE", use_container_width=True)
    st.markdown("---")
    
    # File upload
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], help="Upload a CSV file with date and value columns")

if uploaded_file is not None:
    try:
        # Load the raw data
        df_raw = load_csv(uploaded_file)
        
        with st.sidebar:
            st.success(f"✅ Loaded {len(df_raw)} rows")
            
            # Column selection
            st.header("📊 Column Mapping")
            date_column = st.selectbox("📅 Date Column", df_raw.columns, index=0)
            value_column = st.selectbox("📈 Target Column", 
                                       [col for col in df_raw.columns if col != date_column],
                                       index=0)
        
        # Process the data
        df = process_data(df_raw, date_column, value_column)
        
        with st.sidebar:
            st.markdown("---")
            st.header("⚙️ Configuration")
            
            # Data frequency
            freq = st.selectbox("📆 Data Frequency", 
                               ['D', 'W', 'M', 'Q', 'Y'],
                               index=1,
                               help="D=Daily, W=Weekly, M=Monthly, Q=Quarterly, Y=Yearly")
            
            # Forecast settings
            st.subheader("🔮 Forecast Settings")
            forecast_period = st.slider("Forecast Horizon", 1, 52, 12, 
                                       help="Number of periods to forecast ahead")
            
            # Cross-validation settings
            st.subheader("✅ Cross-Validation")
            cv_horizon = st.slider("CV Horizon", 1, 24, 12,
                                  help="Forecast horizon for cross-validation")
            n_windows = st.slider("CV Windows", 2, 8, 4,
                                 help="Number of cross-validation windows")
            
            # Model selection with random seeds
            st.subheader("🤖 Model Selection")
            selected_models = st.multiselect(
                "Choose Models",
                ["LGBM", "Ridge", "KNN", "MLP", "Random Forest"],
                default=["LGBM", "Ridge"],
                help="Select one or more forecasting models"
            )
            
            # Prediction intervals
            st.subheader("📊 Prediction Intervals")
            show_intervals = st.checkbox("Show Prediction Intervals", value=True)
            interval_levels = st.multiselect(
                "Confidence Levels (%)",
                [90, 95, 99],
                default=[90, 95]
            ) if show_intervals else []
            
            st.markdown("---")
            
            # Generate forecast button
            generate_forecast = st.button("🚀 Generate Forecast", type="primary")
        
        # Main content with tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", 
            "🔄 Cross-Validation", 
            "🎯 Forecast Results",
            "📊 Forecast vs Actual",
            "📋 Raw Data"
        ])
        
        with tab1:
            st.header("Historical Data Visualization")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Observations", len(df))
            with col2:
                st.metric("Date Range", f"{df['ds'].min().date()} to {df['ds'].max().date()}")
            with col3:
                st.metric("Mean Value", f"{df['y'].mean():.2f}")
            
            st.markdown("---")
            
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df['ds'], df['y'], linewidth=2, color='#2E86AB')
            ax.fill_between(df['ds'], df['y'], alpha=0.3, color='#2E86AB')
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax.set_title('Historical Time Series Data', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df['y'].describe().to_frame('Statistics'), use_container_width=True)
            with col2:
                # Distribution plot
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df['y'], bins=30, color='#4CAF50', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Value', fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.set_title('Value Distribution', fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
        
        if generate_forecast and selected_models:
            with st.spinner("🔄 Training models and generating forecasts... Please wait."):
                # Model configuration with seeds
                model_mapping = {
                    "LGBM": lgb.LGBMRegressor(verbosity=-1, random_state=42, n_estimators=200),
                    "Ridge": Ridge(random_state=42),
                    "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance'),
                    "MLP": MLPRegressor(hidden_layer_sizes=(200, 100), solver='adam', max_iter=1000, random_state=42),
                    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
                }
                
                models = {name.lower(): model_mapping[name] for name in selected_models}
                
                # Split data for evaluation (52-week holdout)
                threshold_time = df['ds'].max() - pd.Timedelta(weeks=52)
                df_train = df[df['ds'] <= threshold_time]
                df_test = df[df['ds'] > threshold_time]
                
                # Cross-validation
                mlf_cv = MLForecast(
                    models=models,
                    freq=freq,
                    lags=[1, 4, 8, 12],
                    lag_transforms={
                        1: [ExpandingMean()],
                        4: [RollingMean(window_size=4)],
                        8: [RollingMean(window_size=8)],
                        12: [RollingMean(window_size=12)],
                    },
                    date_features=['month', 'week', 'dayofyear']
                )
                
                crossvalidation_df = mlf_cv.cross_validation(
                    df=df_train,
                    h=cv_horizon,
                    n_windows=n_windows,
                    refit=True,
                )
                
                # Main forecasting
                mlf = MLForecast(
                    models=models,
                    freq=freq,
                    lags=[1, 4, 8, 12],
                    lag_transforms={
                        1: [ExpandingMean()],
                        4: [RollingMean(window_size=4)],
                        8: [RollingMean(window_size=8)],
                        12: [RollingMean(window_size=12)],
                    },
                    date_features=['month', 'week', 'dayofyear']
                )
                
                if show_intervals:
                    mlf.fit(
                        df=df_train,
                        prediction_intervals=PredictionIntervals(n_windows=n_windows, h=forecast_period)
                    )
                    forecasts = mlf.predict(forecast_period, level=interval_levels)
                else:
                    mlf.fit(df_train)
                    forecasts = mlf.predict(forecast_period)
                
                # Store in session state
                st.session_state.forecast_generated = True
                st.session_state.forecasts = forecasts
                st.session_state.crossvalidation_df = crossvalidation_df
                st.session_state.df_test = df_test
                st.session_state.df_train = df_train
                st.session_state.models = models
                
                # Evaluate on test set
                test_results = df_test.merge(forecasts, how='left', on=['unique_id', 'ds'])
                st.session_state.test_results = test_results
                
                # Create comparison table
                comparison_table = create_forecast_vs_actual_table(df_test, forecasts, models)
                st.session_state.comparison_table = comparison_table
                
                # Evaluation metrics
                metrics = [mae, rmse, mape, smape]
                evaluation_results = evaluate(
                    df=test_results,
                    metrics=metrics,
                    models=list(models.keys())
                )
                st.session_state.evaluation_results = evaluation_results
                
                # CV evaluation
                cv_evaluation = evaluate_crossvalidation(crossvalidation_df, metrics, models)
                st.session_state.cv_evaluation = cv_evaluation
                
            st.success("✅ Forecast generated successfully!")
            st.rerun()
        
        # Display results if available
        if st.session_state.forecast_generated:
            with tab2:
                st.header("Cross-Validation Analysis")
                
                # CV Performance metrics
                st.subheader("📊 Cross-Validation Performance Metrics")
                cv_eval_df = st.session_state.cv_evaluation.copy()
                numeric_cols = cv_eval_df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    cv_eval_styled = cv_eval_df.style.background_gradient(
                        cmap='RdYlGn_r', axis=1, subset=numeric_cols
                    ).format("{:.3f}", subset=numeric_cols)
                    st.dataframe(cv_eval_styled, use_container_width=True)
                else:
                    st.dataframe(cv_eval_df, use_container_width=True)
                
                st.markdown("---")
                
                # CV plots
                st.subheader("📈 Cross-Validation Forecasts")
                cutoffs = st.session_state.crossvalidation_df.query('unique_id == "series"')['cutoff'].unique()
                
                n_cutoffs = len(cutoffs)
                fig, axes = plt.subplots(nrows=n_cutoffs, ncols=1, figsize=(14, 5*n_cutoffs), 
                                        gridspec_kw=dict(hspace=0.4))
                
                if n_cutoffs == 1:
                    axes = [axes]
                
                colors = plt.cm.Set2(np.linspace(0, 1, len(st.session_state.models)))
                
                for idx, (cutoff, ax) in enumerate(zip(cutoffs, axes)):
                    max_date = st.session_state.crossvalidation_df.query(
                        'unique_id == "series" & cutoff == @cutoff'
                    )['ds'].max()
                    
                    # Plot training data
                    train_data = st.session_state.df_train[
                        st.session_state.df_train['ds'] < max_date
                    ].query('unique_id == "series"').tail(24)
                    
                    ax.plot(train_data['ds'], train_data['y'], 
                           label='Actual', color='black', linewidth=2, marker='o', markersize=4)
                    
                    # Plot forecasts
                    for model_idx, model in enumerate(st.session_state.models.keys()):
                        cv_data = st.session_state.crossvalidation_df.query(
                            'unique_id == "series" & cutoff == @cutoff'
                        )
                        ax.plot(cv_data['ds'], cv_data[model], 
                               label=f'{model.upper()}', 
                               color=colors[model_idx], linewidth=2, marker='s', markersize=4)
                    
                    ax.set_title(f'Cross-Validation Window {idx+1} | Cutoff: {cutoff}', 
                                fontweight='bold', fontsize=12)
                    ax.set_xlabel('Date', fontweight='bold')
                    ax.set_ylabel('Value', fontweight='bold')
                    ax.legend(loc='best', frameon=True, shadow=True)
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with tab3:
                st.header("Forecast Results & Visualization")
                
                # Test set performance
                st.subheader("📊 Test Set Performance Metrics")
                eval_df = st.session_state.evaluation_results.copy()
                numeric_cols = eval_df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    eval_styled = eval_df.style.background_gradient(
                        cmap='RdYlGn_r', axis=1, subset=numeric_cols
                    ).format("{:.3f}", subset=numeric_cols)
                    st.dataframe(eval_styled, use_container_width=True)
                else:
                    st.dataframe(eval_df, use_container_width=True)
                
                st.markdown("---")
                
                # Forecast visualization
                st.subheader("📈 Forecast Visualization with Prediction Intervals")
                
                fig, ax = plt.subplots(figsize=(16, 8))
                
                # Plot historical data
                ax.plot(st.session_state.df_train['ds'], st.session_state.df_train['y'], 
                       label='Training Data', alpha=0.6, linewidth=2, color='gray')
                ax.plot(st.session_state.df_test['ds'], st.session_state.df_test['y'], 
                       label='Test Data (Actual)', linewidth=2.5, color='black', marker='o', markersize=5)
                
                # Plot predictions
                colors = plt.cm.Set1(np.linspace(0, 1, len(st.session_state.models)))
                
                for model_idx, model in enumerate(st.session_state.models.keys()):
                    ax.plot(st.session_state.forecasts['ds'], st.session_state.forecasts[model], 
                           label=f'{model.upper()} Forecast', linewidth=2.5, 
                           color=colors[model_idx], marker='s', markersize=5, linestyle='--')
                    
                    # Plot prediction intervals
                    if show_intervals:
                        for level_idx, level in enumerate(interval_levels):
                            lower_col = f'{model}_lower_{level}'
                            upper_col = f'{model}_upper_{level}'
                            if lower_col in st.session_state.forecasts.columns:
                                alpha_val = 0.15 - (level_idx * 0.03)
                                ax.fill_between(
                                    st.session_state.forecasts['ds'],
                                    st.session_state.forecasts[lower_col],
                                    st.session_state.forecasts[upper_col],
                                    alpha=alpha_val,
                                    color=colors[model_idx],
                                    label=f'{model.upper()} {level}% CI' if level_idx == 0 else ''
                                )
                
                ax.set_xlabel('Date', fontsize=14, fontweight='bold')
                ax.set_ylabel('Value', fontsize=14, fontweight='bold')
                ax.set_title('Forecast Results with Prediction Intervals', 
                            fontsize=16, fontweight='bold', pad=20)
                ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Forecast table
                st.subheader("📋 Detailed Forecast Table")
                st.dataframe(st.session_state.forecasts, use_container_width=True, height=400)
            
            with tab4:
                st.header("🎯 Forecast vs Actual Analysis")
                
                if st.session_state.comparison_table is not None:
                    comparison = st.session_state.comparison_table
                    
                    # Summary metrics by model
                    st.subheader("📊 Model Performance Summary")
                    
                    summary_data = []
                    for model in st.session_state.models.keys():
                        model_metrics = {
                            'Model': model.upper(),
                            'Mean Residual': comparison[f'{model}_residual'].mean(),
                            'Std Residual': comparison[f'{model}_residual'].std(),
                            'Mean APE (%)': comparison[f'{model}_APE'].mean(),
                            'RMSE': np.sqrt((comparison[f'{model}_residual']**2).mean())
                        }
                        summary_data.append(model_metrics)
                    
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Format summary with safe column detection
                    format_dict = {
                        'Mean Residual': '{:.3f}',
                        'Std Residual': '{:.3f}',
                        'Mean APE (%)': '{:.2f}',
                        'RMSE': '{:.3f}'
                    }
                    
                    # Only format columns that exist
                    existing_format = {k: v for k, v in format_dict.items() if k in summary_df.columns}
                    
                    st.dataframe(
                        summary_df.style.background_gradient(
                            cmap='RdYlGn_r', 
                            subset=[col for col in ['RMSE', 'Mean APE (%)'] if col in summary_df.columns]
                        ).format(existing_format),
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    
                    # Detailed comparison table
                    st.subheader("📋 Detailed Forecast vs Actual Table")
                    
                    # Prepare display columns
                    display_cols = ['ds', 'y']
                    for model in st.session_state.models.keys():
                        display_cols.extend([model, f'{model}_residual', f'{model}_APE'])
                    
                    comparison_display = comparison[display_cols].copy()
                    comparison_display.columns = ['Date', 'Actual'] + [
                        col.replace('_', ' ').title() for col in display_cols[2:]
                    ]
                    
                    # Safe formatting for display
                    numeric_display_cols = [col for col in comparison_display.columns if col != 'Date']
                    format_dict = {col: '{:.3f}' for col in numeric_display_cols}
                    
                    styled_comparison = comparison_display.style.format(format_dict)
                    
                    # Add gradient only to residual and APE columns
                    gradient_cols = [col for col in comparison_display.columns 
                                   if 'Residual' in col or 'Ape' in col]
                    if gradient_cols:
                        styled_comparison = styled_comparison.background_gradient(
                            cmap='RdYlGn_r',
                            subset=gradient_cols
                        )
                    
                    st.dataframe(styled_comparison, use_container_width=True, height=500)
                    
                    # Download button
                    st.markdown("---")
                    csv = comparison.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast vs Actual (CSV)",
                        data=csv,
                        file_name=f"forecast_vs_actual_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Residual plots
                    st.markdown("---")
                    st.subheader("📊 Residual Analysis")
                    
                    n_models = len(st.session_state.models)
                    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
                    if n_models == 1:
                        axes = [axes]
                    
                    for idx, (model, ax) in enumerate(zip(st.session_state.models.keys(), axes)):
                        residuals = comparison[f'{model}_residual']
                        ax.hist(residuals, bins=20, color='#4CAF50', alpha=0.7, edgecolor='black')
                        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Line')
                        ax.set_xlabel('Residual', fontweight='bold')
                        ax.set_ylabel('Frequency', fontweight='bold')
                        ax.set_title(f'{model.upper()} Residuals', fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Generate a forecast to see comparison analysis")
        
        with tab5:
            st.header("Raw Dataset")
            st.dataframe(df, use_container_width=True, height=600)
            
            # Download raw data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Data (CSV)",
                data=csv,
                file_name=f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.exception(e)

else:
    # Landing page
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2 style='color: #2E86AB;'>Welcome to AISPIRE Forecasting Platform</h2>
        <p style='font-size: 1.2rem; color: #555;'>
            A comprehensive time series forecasting solution with advanced analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🤖 Multiple Models</h3>
            <p>Choose from LGBM, Ridge, KNN, MLP, and Random Forest</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>✅ Cross-Validation</h3>
            <p>Robust model evaluation with time-series CV</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>📊 Advanced Analytics</h3>
            <p>Prediction intervals, residual analysis & more</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👆 Please upload a CSV file in the sidebar to begin forecasting")
    
    st.markdown("""
    ### 📖 Getting Started
    
    1. **Upload Data**: Upload your CSV file with date and value columns
    2. **Configure Settings**: Select data frequency, forecast horizon, and models
    3. **Generate Forecast**: Click the button to train models and generate predictions
    4. **Analyze Results**: Explore cross-validation, forecasts, and detailed comparisons
    
    ### 📋 Requirements
    - CSV file with at least two columns (date and numeric values)
    - Date column should be in a standard date format
    - Value column should contain numeric data
    """)
