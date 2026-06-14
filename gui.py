"""
Car Price Prediction — Interactive Cross-Platform GUI
Run: python gui.py
Works on Desktop (VS Code) and Android (Pydroid 3).
"""

import os
import sys
import warnings
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  1. LOAD MODEL BUNDLE
# ══════════════════════════════════════════════════════════════════════════════
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "car_price_model.pkl")

def load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None, f"❌ Model file not found at:\n{MODEL_PATH}\n\nPlease run train_model.py first."
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle, None
    except Exception as e:
        return None, f"❌ Error loading model:\n{str(e)}"

bundle, err_msg = load_bundle()

# ══════════════════════════════════════════════════════════════════════════════
#  2. PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def predict_price(car_name, year, present_price, kms, owner, fuel, seller, transmission):
    if bundle is None:
        return None, err_msg

    try:
        CURRENT_YEAR = bundle.get("current_year", 2026)
        encoders = bundle["encoders"]
        model = bundle["model"]
        best_name = bundle["best_name"]

        # Validation
        if not car_name:
            return None, "⚠️ Please select a car name."
        if not (1995 <= year <= CURRENT_YEAR):
            return None, f"⚠️ Year must be between 1995 and {CURRENT_YEAR}."
        if present_price <= 0:
            return None, "⚠️ Present Price must be greater than 0."
        if kms < 0:
            return None, "⚠️ KM Driven cannot be negative."

        # Check if car_name is known to encoder
        known_cars = list(encoders["car_name"].classes_)
        if car_name not in known_cars:
            return None, f"⚠️ Car '{car_name}' was not seen during training."

        # Feature Engineering
        car_age = CURRENT_YEAR - year
        car_enc = int(encoders["car_name"].transform([car_name])[0])
        fuel_enc = int(encoders["fuel"].transform([fuel])[0])
        seller_enc = int(encoders["seller"].transform([seller])[0])
        trans_enc = int(encoders["transmission"].transform([transmission])[0])

        # Predict — feature order must match train_model.py
        X = np.array([[car_enc, car_age, present_price, kms, fuel_enc, seller_enc, trans_enc, owner]])
        price = float(model.predict(X)[0])

        if price < 0:
            price = 0.0

        # Create Plotly Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=price,
            number=dict(suffix=" L", font=dict(size=36, color="#F97316", family="sans-serif")),
            gauge=dict(
                axis=dict(range=[0, max(20, price * 1.5)], tickcolor="#94A3B8"),
                bar=dict(color="#F97316"),
                bgcolor="#334155",
                bordercolor="rgba(0,0,0,0)",
                steps=[
                    dict(range=[0, price * 0.5], color="#0F172A"),
                    dict(range=[price * 0.5, price], color="rgba(249,115,22,0.2)"),
                ],
            )
        ))
        gauge.update_layout(
            paper_bgcolor="#1E293B",
            font=dict(color="#F1F5F9", family="sans-serif"),
            margin=dict(l=20, r=20, t=40, b=10),
            height=250
        )

        # Format Result Card HTML
        result_html = f"""
        <div style="background: linear-gradient(135deg, rgba(249,115,22,0.12), rgba(251,191,36,0.06)); 
                    border: 1px solid rgba(249,115,22,0.35); border-radius: 14px; padding: 24px; text-align: center; margin-top: 10px;">
            <div style="font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; color: #94A3B8;">Estimated Resale Value</div>
            <div style="font-size: 1rem; color: #CBD5E1; margin: 6px 0 2px;">
                {car_name} &nbsp;·&nbsp; {year} &nbsp;·&nbsp; {fuel}
            </div>
            <div style="font-size: 3rem; font-weight: bold; color: #F97316; line-height: 1.1; margin: 10px 0;">
                <span style="font-size: 1.5rem; vertical-align: super; color: #FBBF24;">₹</span>{price:.2f}
            </div>
            <div style="font-size: 0.9rem; color: #94A3B8;">Lakhs &nbsp;·&nbsp; ≈ ₹ {price*100000:,.0f}</div>
            <div style="display: inline-block; margin-top: 12px; background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); 
                        color: #22C55E; font-size: 0.75rem; padding: 4px 14px; border-radius: 99px;">
                via {best_name}
            </div>
        </div>
        """
        return gauge, result_html

    except Exception as e:
        return None, f"⚠️ Prediction error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
#  3. ACCURACY DASHBOARD LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def get_accuracy_dashboard():
    if bundle is None:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), err_msg

    results = bundle["results"]
    best_name = bundle["best_name"]
    names = list(results.keys())
    colors = ["#F97316" if n == best_name else "#334155" for n in names]

    # 1. Comparison Bar Chart
    fig_compare = make_subplots(rows=1, cols=3, subplot_titles=["R² Score ↑", "MAE (lakhs) ↓", "RMSE (lakhs) ↓"])
    fig_compare.add_trace(go.Bar(x=names, y=[results[n]["r2"] for n in names], marker_color=colors, name="R²"), row=1, col=1)
    fig_compare.add_trace(go.Bar(x=names, y=[results[n]["mae"] for n in names], marker_color=colors, name="MAE"), row=1, col=2)
    fig_compare.add_trace(go.Bar(x=names, y=[results[n]["rmse"] for n in names], marker_color=colors, name="RMSE"), row=1, col=3)
    fig_compare.update_layout(
        paper_bgcolor="#1E293B", plot_bgcolor="#0F172A", font=dict(color="#F1F5F9", size=12),
        margin=dict(l=20, r=20, t=40, b=20), height=300, showlegend=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"), yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
    )

    # 2. Actual vs Predicted Scatter
    y_test = results[best_name]["y_test"]
    y_pred = results[best_name]["y_pred"]
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", marker=dict(color="#F97316", size=6, opacity=0.7)))
    fig_scatter.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#22C55E", width=2, dash="dash")))
    fig_scatter.update_layout(
        paper_bgcolor="#1E293B", plot_bgcolor="#0F172A", font=dict(color="#F1F5F9", size=12),
        margin=dict(l=20, r=20, t=40, b=20), height=300,
        title=f"Actual vs Predicted — {best_name}",
        xaxis_title="Actual (lakhs)", yaxis_title="Predicted (lakhs)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"), yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
    )

    # 3. Residual Histogram
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Histogram(x=y_pred - y_test, nbinsx=30, marker_color="#38BDF8", opacity=0.8))
    fig_resid.add_vline(x=0, line_color="#22C55E", line_width=2, line_dash="dash")
    fig_resid.update_layout(
        paper_bgcolor="#1E293B", plot_bgcolor="#0F172A", font=dict(color="#F1F5F9", size=12),
        margin=dict(l=20, r=20, t=40, b=20), height=300, title="Residual Distribution",
        xaxis_title="Error (lakhs)", yaxis_title="Count",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"), yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
    )

    # 4. Table HTML
    rows = ""
    for n in sorted(results, key=lambda k: results[k]["r2"], reverse=True):
        r = results[n]
        tr_style = "color: #F97316; font-weight: bold;" if n == best_name else "color: #F1F5F9;"
        star = " ★" if n == best_name else ""
        rows += f"<tr style='{tr_style}'><td style='padding:10px;'>{n}{star}</td><td style='padding:10px;'>{r['r2']:.4f}</td><td style='padding:10px;'>{r['mae']:.4f}</td><td style='padding:10px;'>{r['rmse']:.4f}</td></tr>"

    table_html = f"""
    <div style="margin-top: 20px; overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem; color: #F1F5F9;">
            <thead><tr style="border-bottom: 1px solid rgba(255,255,255,0.1); text-align: left;">
                <th style="padding: 10px;">Algorithm</th><th style="padding: 10px;">R²</th><th style="padding: 10px;">MAE</th><th style="padding: 10px;">RMSE</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        <p style="font-size: 0.75rem; color: #94A3B8; margin-top: 10px;">★ Best model selected automatically</p>
    </div>
    """
    return fig_compare, fig_scatter, fig_resid, table_html


# ══════════════════════════════════════════════════════════════════════════════
#  4. GRADIO UI CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
custom_css = """
body, .gradio-container { background-color: #0F172A !important; color: #F1F5F9 !important; font-family: 'Segoe UI', system-ui, sans-serif !important; }
.gr-box { background-color: #1E293B !important; border: 1px solid #334155 !important; border-radius: 12px !important; }
.gr-input, .gr-dropdown { background-color: #334155 !important; color: #F1F5F9 !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 8px !important; height: auto !important; min-height: 38px !important; box-sizing: border-box !important; }
.gr-button-primary { background: linear-gradient(135deg, #F97316, #FBBF24) !important; color: #000 !important; font-weight: bold !important; border: none !important; }
.gr-tabs { border: none !important; }
.tab-nav { background-color: #1E293B !important; border-bottom: 1px solid #334155 !important; }
.tab-nav button { color: #94A3B8 !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
.tab-nav button.selected { color: #F97316 !important; border-bottom: 3px solid #F97316 !important; background: rgba(249,115,22,0.05) !important; }
"""

with gr.Blocks(theme=gr.themes.Default(), css=custom_css, title="Car Price Predictor") as demo:

    # Hero Banner
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #0F172A 0%, #1c1445 50%, #0f2d1a 100%); 
                padding: 30px; border-radius: 16px; border-bottom: 1px solid rgba(249,115,22,0.25); margin-bottom: 20px; text-align: center;">
        <h1 style="font-size: 2.5rem; font-weight: 800; color: #F97316; margin: 0; letter-spacing: 2px;">CAR PRICE PREDICTOR</h1>
        <p style="color: #94A3B8; margin-top: 8px; font-size: 0.9rem; letter-spacing: 1px; text-transform: uppercase;">Machine Learning · Regression · Interactive</p>
    </div>
    """)

    with gr.Tabs():
        # ── TAB 1: PREDICT ──
        with gr.TabItem("⚡ Predict Price"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🚗 Vehicle Identity")
                    car_name_opts = sorted(bundle["car_names"]) if bundle else []
                    in_car_name = gr.Dropdown(
                        choices=car_name_opts,
                        value=car_name_opts[0] if car_name_opts else None,
                        label="Car Name",
                        info="Select the vehicle model from the list"
                    )
                    in_year = gr.Slider(minimum=1995, maximum=2026, value=2018, step=1, label="Manufacturing Year")
                    in_present = gr.Number(value=8.5, label="Current Showroom Price (₹ lakhs)", minimum=0.1)
                    in_kms = gr.Number(value=40000, label="Kilometers Driven", minimum=0)
                    in_owner = gr.Slider(minimum=0, maximum=3, value=0, step=1, label="Previous Owners (0=First, 1=Second, etc.)")

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Specifications")
                    fuel_opts = list(bundle["encoders"]["fuel"].classes_) if bundle else ["Petrol", "Diesel", "CNG"]
                    seller_opts = list(bundle["encoders"]["seller"].classes_) if bundle else ["Individual", "Dealer"]
                    trans_opts = list(bundle["encoders"]["transmission"].classes_) if bundle else ["Manual", "Automatic"]

                    in_fuel = gr.Dropdown(choices=fuel_opts, value=fuel_opts[0] if fuel_opts else "Petrol", label="Fuel Type")
                    in_seller = gr.Dropdown(choices=seller_opts, value=seller_opts[0] if seller_opts else "Individual", label="Seller Type")
                    in_trans = gr.Dropdown(choices=trans_opts, value=trans_opts[0] if trans_opts else "Manual", label="Transmission")

            btn_predict = gr.Button("⚡ PREDICT PRICE", variant="primary", size="lg")

            with gr.Row():
                with gr.Column():
                    out_gauge = gr.Plot(label="Price Gauge")
                with gr.Column():
                    out_result = gr.HTML()

            btn_predict.click(
                fn=predict_price,
                inputs=[in_car_name, in_year, in_present, in_kms, in_owner, in_fuel, in_seller, in_trans],
                outputs=[out_gauge, out_result]
            )

        # ── TAB 2: ACCURACY ──
        with gr.TabItem("📊 Accuracy & Insights"):
            gr.Markdown("### 📈 Model Accuracy Analysis")
            out_compare = gr.Plot(label="Model Comparison")
            out_scatter = gr.Plot(label="Actual vs Predicted")
            out_resid = gr.Plot(label="Residual Distribution")
            out_table = gr.HTML()

    # Load accuracy data on startup
    demo.load(
        fn=get_accuracy_dashboard,
        inputs=[],
        outputs=[out_compare, out_scatter, out_resid, out_table]
    )

# ══════════════════════════════════════════════════════════════════════════════
#  5. LAUNCH (Cross-Platform: Desktop & Android)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 Starting Car Price Predictor GUI...")
    print("-" * 50)

    is_android = 'ANDROID_ROOT' in os.environ or hasattr(sys, 'getandroidapilevel')

    if is_android:
        print("📱 Android environment detected (Pydroid 3).")
        print("📌 Open your mobile browser and navigate to:")
        print("👉 http://127.0.0.1:7860")
        print("💡 Note: If the page doesn't load, try http://localhost:7860")
        print("-" * 50)
        demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=False, share=False, show_error=True)
    else:
        print("💻 Desktop environment detected.")
        print("📌 If it doesn't open automatically, copy this URL into your browser:")
        print("👉 http://127.0.0.1:7860")
        print("-" * 50)
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, share=False, show_error=True)