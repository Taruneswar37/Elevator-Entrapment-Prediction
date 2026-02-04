import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import os, time

from sensor_simulator import ElevatorSensor
from predictor import predict_live

BASE = os.path.dirname(__file__)

# ✅ PROPER Render-safe log location
LOG_DIR = os.path.join(BASE, "logs")
LOG = os.path.join(LOG_DIR, "prediction_history.csv")

# 👉 FORCE folder creation on startup
os.makedirs(LOG_DIR, exist_ok=True)


sensor = ElevatorSensor()

app = dash.Dash(__name__)
app.title = "Elevator – Real Time Safety"

def save_log(data):
    try:
        df = pd.DataFrame([data])

        # ensure directory always exists
        os.makedirs(LOG_DIR, exist_ok=True)

        if not os.path.exists(LOG):
            df.to_csv(LOG, index=False)
        else:
            df.to_csv(LOG, mode='a', header=False, index=False)

    except Exception as e:
        print("LOG SAVE ERROR:", e)


def load_trend():
    try:
        if os.path.exists(LOG):
            df = pd.read_csv(LOG).tail(20)
            if not df.empty:
                return df
    except:
        pass

    # fallback empty structure
    return pd.DataFrame(columns=[
        "time","risk","vibration","temperature"
    ])


def gauge(v):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=v*100,
        title={'text': "Risk %"},
        gauge={
            'axis': {'range':[0,100]},
            'steps':[
                {'range':[0,40],'color':'lightgreen'},
                {'range':[40,70],'color':'orange'},
                {'range':[70,100],'color':'red'}
            ]
        }
    ))

app.layout = html.Div([

    html.H1("🛗 KONE Elevator – Real Time Entrapment Monitor"),

    dcc.Interval(id='timer', interval=4000),

    dcc.Graph(id='gauge'),

    html.Div(id='status'),

    dcc.Graph(id='trend'),

    dcc.Graph(id='raw')

])

@app.callback(
    [Output('gauge','figure'),
     Output('status','children'),
     Output('trend','figure'),
     Output('raw','figure')],
    [Input('timer','n_intervals')]
)
def update(_):

    data = sensor.next()

    proba, risk = predict_live(data)

    data["risk"] = proba
    data["time"] = time.strftime("%H:%M:%S")

    save_log(data)

    status = "✅ SAFE"
    color="green"
    if risk:
        status="🚨 HIGH RISK – CALL MAINTENANCE"
        color="red"

    df = load_trend()

    trend = go.Figure()
    trend.add_trace(go.Scatter(
        x=df["time"],
        y=df["risk"],
        mode='lines+markers',
        name='Risk'
    ))
    trend.update_layout(title="Real Model Predictions")

    raw = go.Figure()
    raw.add_trace(go.Scatter(y=df["vibration"], name="vibration"))
    raw.add_trace(go.Scatter(y=df["temperature"], name="temp"))
    raw.update_layout(title="Sensor Behaviour")

    return gauge(proba), html.H3(status,style={'color':color}), trend, raw


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

