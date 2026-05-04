import optuna
import optuna.visualization as vis
import os

def analyze_my_sweep():
    print("Loading database...")
    study = optuna.load_study(
        study_name="quant_transformer_sweep", 
        storage="sqlite:///optuna_sweep.db"
    )

    print("\n" + "="*50)
    print("🏆 CURRENT BEST RESULTS 🏆")
    print(f"Best Test Accuracy: {-study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # ---------------------------------------------------------
    # GENERATE 100% OFFLINE HTML CHARTS
    # ---------------------------------------------------------
    print("\nGenerating OFFLINE HTML charts...")
    
    try:
        # 1. Parameter Importances
        fig1 = vis.plot_param_importances(study)
        # include_plotlyjs=True forces the JS engine to live inside the HTML file!
        fig1.write_html("chart_1_importances.html", include_plotlyjs=True)
        print(" -> Saved 'chart_1_importances.html'")

        # 2. Optimization History
        fig2 = vis.plot_optimization_history(study)
        fig2.write_html("chart_2_history.html", include_plotlyjs=True)
        print(" -> Saved 'chart_2_history.html'")
        
        # 3. Slice Plot
        fig3 = vis.plot_slice(study)
        fig3.write_html("chart_3_slice.html", include_plotlyjs=True)
        print(" -> Saved 'chart_3_slice.html'")
        
        print("\n✅ Success! Open your file explorer and double-click the HTML files to view them.")
        
    except ImportError:
        print("\n[ERROR] To generate the charts, please install plotly:")
        print("pip install plotly")

if __name__ == "__main__":
    analyze_my_sweep()