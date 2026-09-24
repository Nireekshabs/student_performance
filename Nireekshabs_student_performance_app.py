"""
Student Performance Analysis & Prediction Dashboard
====================================================
A complete single-file Dash application that:
  - Loads student_performance_dataset (1).csv
  - Performs exploratory data analysis (EDA)
  - Trains a Random Forest classifier to predict performance_level
  - Visualises key findings via interactive Plotly charts
  - Serves an interactive Dash dashboard on http://127.0.0.1:8050

Run:
    python student_performance_app.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = "student_performance_dataset (1).csv"
df = pd.read_csv(CSV_PATH)

# Basic cleaning
df = df.dropna()
df["enrollment_date"] = pd.to_datetime(df["enrollment_date"])
df["enrollment_month"] = df["enrollment_date"].dt.month_name()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING  &  MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = ["age", "daily_study_hours", "attendance_percent",
            "exam_score", "study_efficiency"]

le_course = LabelEncoder()
le_city   = LabelEncoder()
df["course_enc"] = le_course.fit_transform(df["course"])
df["city_enc"]   = le_city.fit_transform(df["city"])

le_target = LabelEncoder()
df["perf_enc"] = le_target.fit_transform(df["performance_level"])
# Classes: Excellent=0, Good=1, Average=2, Poor=3  (alphabetical order by sklearn)

X = df[FEATURES + ["course_enc", "city_enc"]]
y = df["perf_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=150, max_depth=8,
                               random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

y_pred   = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

# Feature importance
feat_imp = pd.DataFrame({
    "Feature":    FEATURES + ["course_enc", "city_enc"],
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

# Confusion matrix
classes = le_target.classes_
cm       = confusion_matrix(y_test, y_pred)

# Classification report → dict
cr = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SUMMARY STATS  (printed to console on startup)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  STUDENT PERFORMANCE ANALYSIS  –  Summary")
print("=" * 60)
print(f"  Total students      : {len(df)}")
print(f"  Features used       : {', '.join(FEATURES)}")
print(f"  Model Accuracy      : {accuracy}%")
print(f"  Performance levels  : {list(classes)}")
print("=" * 60)
print(df[["age","daily_study_hours","attendance_percent",
          "exam_score","study_efficiency"]].describe().round(2).to_string())
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold
PERF_COLOR = {
    "Excellent": "#2ecc71",
    "Good":      "#3498db",
    "Average":   "#f39c12",
    "Poor":      "#e74c3c",
}

def make_perf_pie():
    counts = df["performance_level"].value_counts().reset_index()
    counts.columns = ["Level", "Count"]
    fig = px.pie(counts, names="Level", values="Count",
                 title="Performance Level Distribution",
                 color="Level", color_discrete_map=PERF_COLOR,
                 hole=0.45)
    fig.update_traces(textinfo="percent+label", pull=[0.04]*len(counts))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=50, b=10))
    return fig

def make_exam_hist():
    fig = px.histogram(df, x="exam_score", color="performance_level",
                       nbins=20, barmode="overlay",
                       title="Exam Score Distribution by Performance Level",
                       color_discrete_map=PERF_COLOR, opacity=0.75,
                       labels={"exam_score": "Exam Score",
                               "performance_level": "Level"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=50, b=10))
    return fig

def make_scatter():
    fig = px.scatter(df, x="daily_study_hours", y="exam_score",
                     color="performance_level", size="attendance_percent",
                     hover_data=["name", "course"],
                     title="Study Hours vs Exam Score",
                     color_discrete_map=PERF_COLOR,
                     labels={"daily_study_hours": "Daily Study Hours",
                             "exam_score": "Exam Score",
                             "performance_level": "Level"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=50, b=10))
    return fig

def make_course_bar():
    grp = df.groupby(["course", "performance_level"]).size().reset_index(name="count")
    fig = px.bar(grp, x="course", y="count", color="performance_level",
                 barmode="group", title="Performance by Course",
                 color_discrete_map=PERF_COLOR,
                 labels={"course": "Course", "count": "# Students",
                         "performance_level": "Level"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=50, b=10))
    return fig

def make_city_bar():
    grp = df.groupby("city")["exam_score"].mean().reset_index()
    grp.columns = ["city", "avg_score"]
    grp.sort_values("avg_score", ascending=False, inplace=True)
    fig = px.bar(grp, x="city", y="avg_score",
                 title="Average Exam Score by City",
                 color="avg_score", color_continuous_scale="Blues",
                 labels={"city": "City", "avg_score": "Avg Exam Score"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False,
                      margin=dict(t=50, b=10))
    return fig

def make_feat_imp_bar():
    fig = px.bar(feat_imp, x="Importance", y="Feature",
                 orientation="h",
                 title="Feature Importances (Random Forest)",
                 color="Importance", color_continuous_scale="Teal")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False,
                      margin=dict(t=50, b=10))
    return fig

def make_confusion_matrix():
    fig = px.imshow(cm, text_auto=True,
                    x=list(classes), y=list(classes),
                    color_continuous_scale="Blues",
                    title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=50, b=10))
    return fig

def make_attendance_box():
    fig = px.box(df, x="performance_level", y="attendance_percent",
                 color="performance_level",
                 title="Attendance % by Performance Level",
                 color_discrete_map=PERF_COLOR,
                 labels={"performance_level": "Level",
                         "attendance_percent": "Attendance %"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False,
                      margin=dict(t=50, b=10))
    return fig

def make_corr_heatmap():
    num_cols = ["age","daily_study_hours","attendance_percent",
                "exam_score","study_efficiency"]
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True,
                    color_continuous_scale="RdBu",
                    title="Correlation Heatmap",
                    zmin=-1, zmax=1)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=50, b=10))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 5.  DASH APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY],
                title="Student Performance Dashboard")

# KPI cards
def kpi_card(title, value, color="primary"):
    return dbc.Card([
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "13px"}),
            html.H4(str(value), className=f"text-{color} fw-bold mb-0"),
        ])
    ], className="shadow-sm border-0 rounded-3")

total_students    = len(df)
avg_exam          = round(df["exam_score"].mean(), 1)
avg_attendance    = round(df["attendance_percent"].mean(), 1)
excellent_pct     = round((df["performance_level"] == "Excellent").mean() * 100, 1)

app.layout = dbc.Container(fluid=True, children=[

    # ── Header ──
    dbc.Row([
        dbc.Col(html.Div([
            html.H2("🎓 Student Performance Dashboard",
                    className="mb-0 fw-bold text-white"),
            html.P("Data Analysis · Prediction · Insights",
                   className="text-white-50 mb-0"),
        ], className="p-4 rounded-3",
           style={"background": "linear-gradient(135deg,#1a73e8,#0d47a1)"}),
        width=12)
    ], className="mb-4 mt-2"),

    # ── KPI Cards ──
    dbc.Row([
        dbc.Col(kpi_card("Total Students",    total_students,  "primary"),  md=3),
        dbc.Col(kpi_card("Avg Exam Score",    avg_exam,        "success"),  md=3),
        dbc.Col(kpi_card("Avg Attendance %",  avg_attendance,  "warning"),  md=3),
        dbc.Col(kpi_card("Model Accuracy",    f"{accuracy}%",  "danger"),   md=3),
    ], className="mb-4 g-3"),

    # ── Tabs ──
    dbc.Tabs(id="tabs", active_tab="tab-overview", children=[

        # ──── TAB 1 : Overview ────
        dbc.Tab(label="📊 Overview", tab_id="tab-overview", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=make_perf_pie(),    id="pie"),    md=5),
                dbc.Col(dcc.Graph(figure=make_exam_hist(),   id="hist"),   md=7),
            ], className="mt-3 g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=make_attendance_box(), id="box"), md=6),
                dbc.Col(dcc.Graph(figure=make_corr_heatmap(),   id="corr"),md=6),
            ], className="mt-3 g-3"),
        ]),

        # ──── TAB 2 : Deep Dive ────
        dbc.Tab(label="🔍 Deep Dive", tab_id="tab-dive", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=make_scatter(),    id="scatter"), md=7),
                dbc.Col(dcc.Graph(figure=make_course_bar(), id="course"),  md=5),
            ], className="mt-3 g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=make_city_bar(),   id="city"),    md=12),
            ], className="mt-3 g-3"),
        ]),

        # ──── TAB 3 : Prediction Model ────
        dbc.Tab(label="🤖 Prediction Model", tab_id="tab-model", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=make_feat_imp_bar(),    id="fimp"),  md=6),
                dbc.Col(dcc.Graph(figure=make_confusion_matrix(), id="cmtx"), md=6),
            ], className="mt-3 g-3"),

            # Metric table
            dbc.Row([
                dbc.Col([
                    html.H5("Classification Report", className="mt-3 mb-2 fw-bold"),
                    dash_table.DataTable(
                        id="cr-table",
                        columns=[{"name": c, "id": c} for c in
                                 ["Class","Precision","Recall","F1-Score","Support"]],
                        data=[{
                            "Class":     cls,
                            "Precision": round(cr[cls]["precision"], 3),
                            "Recall":    round(cr[cls]["recall"], 3),
                            "F1-Score":  round(cr[cls]["f1-score"], 3),
                            "Support":   int(cr[cls]["support"]),
                        } for cls in classes],
                        style_header={"backgroundColor":"#1a73e8",
                                      "color":"white","fontWeight":"bold"},
                        style_cell={"textAlign":"center","padding":"8px",
                                    "fontFamily":"sans-serif","fontSize":"13px"},
                        style_data_conditional=[
                            {"if":{"row_index":"odd"},
                             "backgroundColor":"#f0f4ff"}
                        ],
                        style_table={"overflowX":"auto"},
                    ),
                ], md=10),
            ], className="mt-2 g-3"),
        ]),

        # ──── TAB 4 : Predict a Student ────
        dbc.Tab(label="🔮 Predict a Student", tab_id="tab-predict", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Enter Student Details", className="mt-4 fw-bold"),
                    dbc.Label("Age"),
                    dcc.Slider(id="p-age", min=16, max=35, step=1, value=22,
                               marks={16:"16",20:"20",25:"25",30:"30",35:"35"},
                               tooltip={"placement":"bottom"}),
                    dbc.Label("Daily Study Hours", className="mt-3"),
                    dcc.Slider(id="p-study", min=0.5, max=12, step=0.5, value=5,
                               marks={0.5:"0.5",4:"4",8:"8",12:"12"},
                               tooltip={"placement":"bottom"}),
                    dbc.Label("Attendance %", className="mt-3"),
                    dcc.Slider(id="p-attend", min=50, max=100, step=1, value=80,
                               marks={50:"50",70:"70",85:"85",100:"100"},
                               tooltip={"placement":"bottom"}),
                    dbc.Label("Exam Score", className="mt-3"),
                    dcc.Slider(id="p-exam", min=0, max=100, step=1, value=65,
                               marks={0:"0",25:"25",50:"50",75:"75",100:"100"},
                               tooltip={"placement":"bottom"}),
                    dbc.Label("Study Efficiency", className="mt-3"),
                    dcc.Slider(id="p-eff", min=2, max=35, step=0.5, value=12,
                               marks={2:"2",10:"10",20:"20",35:"35"},
                               tooltip={"placement":"bottom"}),
                    dbc.Label("Course", className="mt-3"),
                    dcc.Dropdown(id="p-course",
                                 options=[{"label":c,"value":c}
                                          for c in sorted(df["course"].unique())],
                                 value="Data Science", clearable=False),
                    dbc.Label("City", className="mt-3"),
                    dcc.Dropdown(id="p-city",
                                 options=[{"label":c,"value":c}
                                          for c in sorted(df["city"].unique())],
                                 value="Delhi", clearable=False),
                    dbc.Button("Predict Performance", id="predict-btn",
                               color="primary", className="mt-4 w-100"),
                ], md=5),

                dbc.Col([
                    html.Div(id="predict-result", className="mt-5"),
                ], md=7),
            ], className="mt-2 g-4"),
        ]),

        # ──── TAB 5 : Data Table ────
        dbc.Tab(label="📋 Dataset", tab_id="tab-data", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Filter by Performance Level", className="mt-3 mb-1"),
                    dcc.Dropdown(id="tbl-filter",
                                 options=[{"label":"All","value":"All"}] +
                                         [{"label":v,"value":v}
                                          for v in df["performance_level"].unique()],
                                 value="All", clearable=False,
                                 style={"width":"250px"}),
                ], md=4),
            ]),
            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(
                        id="main-table",
                        columns=[{"name": c, "id": c} for c in
                                 ["student_id","name","age","city","course",
                                  "daily_study_hours","attendance_percent",
                                  "exam_score","performance_level"]],
                        data=df[["student_id","name","age","city","course",
                                 "daily_study_hours","attendance_percent",
                                 "exam_score","performance_level"]].to_dict("records"),
                        page_size=15,
                        sort_action="native",
                        filter_action="native",
                        style_header={"backgroundColor":"#1a73e8",
                                      "color":"white","fontWeight":"bold"},
                        style_cell={"textAlign":"center","padding":"7px",
                                    "fontFamily":"sans-serif","fontSize":"13px"},
                        style_data_conditional=[
                            {"if":{"filter_query":'{performance_level} = "Excellent"'},
                             "backgroundColor":"#d4edda","color":"#155724"},
                            {"if":{"filter_query":'{performance_level} = "Poor"'},
                             "backgroundColor":"#f8d7da","color":"#721c24"},
                        ],
                        style_table={"overflowX":"auto"},
                    ),
                ], md=12),
            ], className="mt-2"),
        ]),

    ]),

    # Footer
    html.Hr(className="mt-5"),
    html.P("Student Performance Analytics  ·  Built with Dash & Plotly  ·  Random Forest Classifier",
           className="text-muted text-center mb-3", style={"fontSize":"12px"}),
])

# ─────────────────────────────────────────────────────────────────────────────
# 6.  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("main-table", "data"),
    Input("tbl-filter", "value"),
)
def filter_table(level):
    if level == "All":
        filtered = df
    else:
        filtered = df[df["performance_level"] == level]
    return filtered[["student_id","name","age","city","course",
                      "daily_study_hours","attendance_percent",
                      "exam_score","performance_level"]].to_dict("records")


@app.callback(
    Output("predict-result", "children"),
    Input("predict-btn", "n_clicks"),
    [
        dash.dependencies.State("p-age",    "value"),
        dash.dependencies.State("p-study",  "value"),
        dash.dependencies.State("p-attend", "value"),
        dash.dependencies.State("p-exam",   "value"),
        dash.dependencies.State("p-eff",    "value"),
        dash.dependencies.State("p-course", "value"),
        dash.dependencies.State("p-city",   "value"),
    ],
    prevent_initial_call=True,
)
def predict(n, age, study, attend, exam, eff, course, city):
    try:
        c_enc = le_course.transform([course])[0]
    except ValueError:
        c_enc = 0
    try:
        ci_enc = le_city.transform([city])[0]
    except ValueError:
        ci_enc = 0

    inp = np.array([[age, study, attend, exam, eff, c_enc, ci_enc]])
    pred_enc   = model.predict(inp)[0]
    pred_prob  = model.predict_proba(inp)[0]
    pred_label = le_target.inverse_transform([pred_enc])[0]

    color_map = {"Excellent":"success","Good":"primary",
                 "Average":"warning","Poor":"danger"}
    badge_color = color_map.get(pred_label, "secondary")

    prob_fig = go.Figure(go.Bar(
        x=[round(p*100,1) for p in pred_prob],
        y=list(le_target.classes_),
        orientation="h",
        marker_color=["#2ecc71","#e74c3c","#3498db","#f39c12"][:len(pred_prob)],
        text=[f"{round(p*100,1)}%" for p in pred_prob],
        textposition="outside",
    ))
    prob_fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Probability (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(t=40, b=10),
    )

    return html.Div([
        html.H5("Prediction Result", className="fw-bold"),
        dbc.Alert([
            html.Span("Predicted Performance Level: ", className="fw-bold"),
            dbc.Badge(pred_label, color=badge_color, className="fs-6 ms-2"),
        ], color=badge_color, className="mt-2"),
        dcc.Graph(figure=prob_fig),
        html.P(f"Tips: Attend ≥85% of classes, study ≥5 hrs/day, and target ≥70 exam score for 'Excellent' performance.",
               className="text-muted mt-2", style={"fontSize":"13px"}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀  Starting dashboard at  http://127.0.0.1:8050\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
