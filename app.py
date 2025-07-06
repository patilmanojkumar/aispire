import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
import os

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
    return evaluations.style.background_gradient(cmap='RdYlGn_r', axis=1)

# Set page config
st.set_page_config(page_title="AISPIRE", layout="wide")

# Title
st.title("AISPIRE")

# File upload and data loading
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

@st.cache_data
def process_data(df, date_column, value_column):
    df = df[[date_column, value_column]]
    df.columns = ['ds', 'y']
    df.insert(0, 'unique_id', 'series')
    df['ds'] = pd.to_datetime(df['ds'])
    # Convert target column to numeric
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    # Remove any rows with NaN values
    df = df.dropna()
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    return df

if uploaded_file is not None:
    try:
        # Load the data
        df_raw = pd.read_csv(uploaded_file)
        
        # Move column selection outside cached function
        date_column = st.selectbox("Select Date Column", df_raw.columns)
        value_column = st.selectbox("Select Target Column", df_raw.columns)
        
        # Process the data using cached function
        df = process_data(df_raw, date_column, value_column)
        success = True
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        success = False

    if success:
        # Sidebar
        st.sidebar.header("Data Settings")
        freq = st.sidebar.selectbox("Select Data Frequency", 
                                  ['D', 'W', 'M', 'Q', 'Y'],
                                  help="D=Daily, W=Weekly, M=Monthly, Q=Quarterly, Y=Yearly")
        
        # Sidebar
        st.sidebar.header("Forecast Settings")
        forecast_period = st.sidebar.slider("Forecast Period (weeks)", 1, 52, 12)
        selected_models = st.sidebar.multiselect(
            "Select Models",
            ["LGBM", "Ridge", "KNN", "MLP", "Random Forest"],
            default=["LGBM", "Ridge"]
        )
        
        # Main content
        st.header("Historical Data")
        fig = plot_series(df)
        st.pyplot(fig)
        
        # Model configuration
        model_mapping = {
            "LGBM": lgb.LGBMRegressor(verbosity=-1),
            "Ridge": Ridge(),
            "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance'),
            "MLP": MLPRegressor(hidden_layer_sizes=(200, 100), solver='adam', max_iter=1000),
            "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10)
        }
        
        # Add to sidebar
        st.sidebar.header("Prediction Interval Settings")
        show_intervals = st.sidebar.checkbox("Show Prediction Intervals", value=True)
        interval_levels = st.sidebar.multiselect(
            "Confidence Levels (%)",
            [90, 95, 99],
            default=[90, 95]
        ) if show_intervals else []
        
        if st.button("Generate Forecast"):
            # Create models dictionary from selected models
            models = {name.lower(): model_mapping[name] for name in selected_models}
            
            # Split data for evaluation
            threshold_time = df['ds'].max() - pd.Timedelta(weeks=52)
            df_train = df[df['ds'] <= threshold_time]
            df_test = df[df['ds'] > threshold_time]

            # Add cross-validation
            mlf_cv = MLForecast(
                models=models,
                freq=freq,
                target_transforms=[Differences([1, 4])],
                lags=[1, 4, 8, 12],
                lag_transforms={
                    1: [ExpandingMean()],
                    4: [RollingMean(window_size=4)],
                    8: [RollingMean(window_size=8)],
                    12: [RollingMean(window_size=12)],
                },
                date_features=['month', 'week', 'dayofyear']
            )

            # Perform cross-validation
            crossvalidation_df = mlf_cv.cross_validation(
                df=df_train,
                h=12,
                n_windows=4,
                refit=True,
            )

            # Plot cross-validation results
            st.header("Cross-Validation Results")
            cutoffs = crossvalidation_df.query('unique_id == "series"')['cutoff'].unique()
            fig, ax = plt.subplots(nrows=len(cutoffs), ncols=1, figsize=(14, 14), 
                                 gridspec_kw=dict(hspace=0.8))
            
            for cutoff, axi in zip(cutoffs, ax.flat):
                max_date = crossvalidation_df.query('unique_id == "series" & cutoff == @cutoff')['ds'].max()
                df_train[df_train['ds'] < max_date].query('unique_id == "series"').tail(24).set_index('ds').plot(
                    ax=axi, title=f'Cutoff: {cutoff}', y='y', label='Actual'
                )
                for m in models.keys():
                    crossvalidation_df.query('unique_id == "series" & cutoff == @cutoff').set_index('ds').plot(
                        ax=axi, y=m, label=m
                    )
                axi.legend()
            
            # After cross-validation plotting
            st.pyplot(fig)
            plt.close()

            # Add cross-validation evaluation
            st.header("Cross-Validation Performance")
            metrics = [mae, rmse, mape, smape]
            cv_evaluation = evaluate_crossvalidation(crossvalidation_df, metrics, models)
            st.dataframe(cv_evaluation)

            # Continue with regular forecasting
            # Create MLForecast instance
            mlf = MLForecast(
                models=models,
                freq=freq,
                target_transforms=[Differences([1, 4])],
                lags=[1, 4, 8, 12],
                lag_transforms={
                    1: [ExpandingMean()],
                    4: [RollingMean(window_size=4)],
                    8: [RollingMean(window_size=8)],
                    12: [RollingMean(window_size=12)],
                },
                date_features=['month', 'week', 'dayofyear']
            )
            
            with st.spinner("Training models and generating forecast..."):
                if show_intervals:
                    mlf.fit(
                        df=df_train,
                        prediction_intervals=PredictionIntervals(n_windows=4, h=forecast_period)
                    )
                    forecasts = mlf.predict(forecast_period, level=interval_levels)
                else:
                    mlf.fit(df_train)
                    forecasts = mlf.predict(forecast_period)
                
                # Display forecast
                st.header("Forecast Results")
                st.dataframe(forecasts)
                
                # Plot forecast with prediction intervals
                plt.figure(figsize=(12, 6))
                
                # Plot historical data
                plt.plot(df_train['ds'], df_train['y'], label='Historical', alpha=0.5)
                
                # Plot predictions for each model
                for model in models.keys():
                    plt.plot(forecasts['ds'], forecasts[model], label=model)
                    
                    # Plot prediction intervals if enabled
                    if show_intervals:
                        for level in interval_levels:
                            lower_col = f'{model}_lower_{level}'
                            upper_col = f'{model}_upper_{level}'
                            if lower_col in forecasts.columns and upper_col in forecasts.columns:
                                plt.fill_between(
                                    forecasts['ds'],
                                    forecasts[lower_col],
                                    forecasts[upper_col],
                                    alpha=0.1,
                                    label=f'{model} {level}% PI'
                                )
                plt.legend()
                plt.title("Price Forecast with Prediction Intervals")
                st.pyplot(plt)
                
                # Model Evaluation
                st.header("Model Performance Metrics")
                metrics = [mae, rmse, mape, smape]
                
                # Evaluate on test set
                test_results = df_test.merge(forecasts, how='left', on=['unique_id', 'ds'])
                evaluation_results = evaluate(
                    df=test_results,
                    metrics=metrics,
                    models=list(models.keys())
                )
                
                # After evaluation on test set...
                st.dataframe(evaluation_results)
                
                # Generate future forecasts using full data
                mlf_future = MLForecast(
                    models=models,
                    freq=freq,
                    target_transforms=[Differences([1, 4])],
                    lags=[1, 4, 8, 12],
                    lag_transforms={
                        1: [ExpandingMean()],
                        4: [RollingMean(window_size=4)],
                        8: [RollingMean(window_size=8)],
                        12: [RollingMean(window_size=12)],
                    },
                    date_features=['month', 'week', 'dayofyear']
                )
                
                # Fit on entire dataset and predict future
                with st.spinner("Generating future forecasts..."):
                    if show_intervals:
                        mlf_future.fit(
                            df=df,
                            prediction_intervals=PredictionIntervals(n_windows=4, h=forecast_period)
                        )
                        future_forecasts = mlf_future.predict(forecast_period, level=interval_levels)
                    else:
                        mlf_future.fit(df)
                        # In the prediction sections, modify the predict calls
                        if selected_exog:
                            future_exog = pd.DataFrame(index=future_forecasts.index)
                            for col in selected_exog:
                                # Here you need to provide the future values for exogenous variables
                                # This is just a simple example using the last known value
                                future_exog[col] = df[col].iloc[-1]
                            future_forecasts = mlf_future.predict(forecast_period, X_df=future_exog)
                        else:
                            future_forecasts = mlf_future.predict(forecast_period)
                
                # Display future forecast
                st.header("Future Forecast Results")
                st.dataframe(future_forecasts)
                
                # Plot future forecasts
                plt.figure(figsize=(12, 6))
                plt.plot(df['ds'], df['y'], label='Historical', alpha=0.5)
                
                for model in models.keys():
                    plt.plot(future_forecasts['ds'], future_forecasts[model], 
                            label=f'{model} Future Forecast', linestyle='--')
                    
                    if show_intervals:
                        for level in interval_levels:
                            lower_col = f'{model}_lower_{level}'
                            upper_col = f'{model}_upper_{level}'
                            if lower_col in future_forecasts.columns and upper_col in future_forecasts.columns:
                                plt.fill_between(
                                    future_forecasts['ds'],
                                    future_forecasts[lower_col],
                                    future_forecasts[upper_col],
                                    alpha=0.1,
                                    label=f'{model} {level}% PI'
                                )
                
                plt.legend()
                plt.title("Future Price Forecast with Prediction Intervals")
                st.pyplot(plt)
    
    # Display raw data
    st.header("Raw Data")
    st.dataframe(df)
else:
    st.info("Please upload a CSV file to begin forecasting. The file should contain at least two columns: one for dates and one for values.")
