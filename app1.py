import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px

# 设置页面配置
st.set_page_config(
    page_title="Power Density Predictor",
    page_icon="🔋",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .feature-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .feature-table {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    /* 自定义输入框样式 */
    .stNumberInput input {
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<div class="main-header">🔋 Power Density Predictor</div>', unsafe_allow_html=True)

# GitHub 配置
GITHUB_USERNAME = "JJJ069"
REPO_NAME = "Power-Density-Predictor"
BRANCH = "main"  # 或 "master"
MODEL_PATH = "Model.pkl"  # 模型在仓库中的路径
SCALER_PATH = "Scaler.pkl"  # 标准化器在仓库中的路径

# 构建 GitHub raw URL
MODEL_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{REPO_NAME}/{BRANCH}/{MODEL_PATH}"
SCALER_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{REPO_NAME}/{BRANCH}/{SCALER_PATH}"

# 模型管理
with st.sidebar:
    st.header("🔧 Model Settings")

    # 自动加载选项
    auto_load = st.checkbox("自动从 GitHub 加载模型", value=True,
                            help="勾选后自动从 GitHub 加载最新模型")

    if auto_load:
        try:
            # 下载并加载模型
            with st.spinner("正在从 GitHub 加载模型..."):
                # 下载模型
                model_response = requests.get(MODEL_URL)
                model_response.raise_for_status()

                # 下载标准化器
                scaler_response = requests.get(SCALER_URL)
                scaler_response.raise_for_status()

                # 加载模型和标准化器
                model = pickle.loads(model_response.content)
                scaler = pickle.loads(scaler_response.content)

                st.success("✅ 模型加载成功！")
                st.info(f"模型来源: {MODEL_URL}")

                # 可选：显示模型信息
                if st.button("显示模型信息"):
                    st.write(f"模型类型: {type(model).__name__}")
                    st.write(f"标准化器类型: {type(scaler).__name__}")

        except Exception as e:
            st.error(f"❌ 加载失败: {str(e)}")
            st.info("请尝试手动上传模型文件")

            # 回退到手动上传
            uploaded_model = st.file_uploader("上传训练好的模型文件",
                                              type=['pkl'],
                                              help="上传训练好的 .pkl 模型文件")

            uploaded_scaler = st.file_uploader("上传数据标准化器",
                                               type=['pkl'],
                                               help="上传对应的 scaler.pkl 文件")
    else:
        # 手动上传模式
        uploaded_model = st.file_uploader("上传训练好的模型文件",
                                          type=['pkl'],
                                          help="上传训练好的 .pkl 模型文件")

        uploaded_scaler = st.file_uploader("上传数据标准化器",
                                           type=['pkl'],
                                           help="上传对应的 scaler.pkl 文件")

# 定义特征配置
feature_config = {
    'LH(kJ/kg)': {
        'default': 225,
        'min': 0,
        'max': 5000,
        'step': 0.001,
        'description': 'Latent heat'
    },
    'MT(°C)': {
        'default': 50,
        'min': 0,
        'max': 500,
        'step': 0.001,
        'description': 'Melt point'
    },
    'TC(W/mK)': {
        'default': 0.2,
        'min': 0.0,
        'max': 10.0,
        'step': 0.001,
        'description': 'Thermal conductivity'
    },
    'CP(kJ/kgK)': {
        'default': 4.18,
        'min': 0.0,
        'max': 10.0,
        'step': 0.001,
        'description': 'Specific heat capacity'
    },
    'Mass(kg)': {
        'default': 1.0,
        'min': 0.0,
        'max': 1000.0,
        'step': 0.001,
        'description': 'Mass'
    },
    'FVR': {
        'default': 0.1,
        'min': 0.0,
        'max': 1.0,
        'step': 0.001,
        'description': 'Fin volume ratio'
    },
    'CCM': {
        'default': 1,
        'min': 0,
        'max': 1,
        'step': 1,
        'description': 'Close-contact melting'
    },
    'TD(°C)': {
        'default': 30.0,
        'min': 0.0,
        'max': 200.0,
        'step': 0.001,
        'description': 'Thermal temperature difference'
    },
    'CD(°C)': {
        'default': 30.0,
        'min': 0.0,
        'max': 200.0,
        'step': 0.001,
        'description': 'Cold temperature difference'
    },
        'HTA(m2)': {
        'default': 1.0,
        'min': 0.0,
        'max': 100.0,
        'step': 0.000001,
        'description': 'Heat transfer area'
    },
    'WTC(W/mK)': {
        'default': 200.0,
        'min': 0.0,
        'max': 2000.0,
        'step': 0.001,
        'description': 'Wall thermal conductivity'
    },
    'FTC(W/mK)': {
        'default': 0.6,
        'min': 0.0,
        'max': 1000.0,
        'step': 0.001,
        'description': 'Fluid thermal conductivity'
    },
    'LPH(L/h)': {
        'default': 100.0,
        'min': 0.0,
        'max': 1000.0,
        'step': 0.001,
        'description': 'litres per hour'
    },
    'AR': {
        'default': 1.0,
        'min': 0.1,
        'max': 100.0,
        'step': 0.001,
        'description': 'Aspect ratio'
    },
    'IA(°)': {
        'default': 0.0,
        'min': 0.0,
        'max': 90.0,
        'step': 0.001,
        'description': 'Inclined angle'
    }
}


# 加载模型
@st.cache_resource
def load_model(uploaded_file):
    """Loading model"""
    try:
        model = joblib.load(uploaded_file)
        st.sidebar.success(f"✅ Model loaded successfully！")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None


# 加载标准化器
@st.cache_resource
def load_scaler(uploaded_file):
    """加载数据标准化器"""
    try:
        scaler = joblib.load(uploaded_file)
        st.sidebar.success(f"✅ Scaler loaded successfully！")
        return scaler
    except Exception as e:
        st.error(f"❌ Failed to load scaler: {str(e)}")
        return None


# 格式化数值为6位小数
def format_value(value):
    """将数值格式化为6位小数"""
    return float(f"{value:.6f}")


# 数据预处理
def preprocess_input(input_features, scaler):
    """对输入数据进行与训练时相同的预处理"""
    try:
        # 创建DataFrame，确保特征顺序与训练时一致
        # 按照模型特征顺序排列
        feature_order = [
            'LH(kJ/kg)', 'MT(°C)', 'TC(W/mK)', 'CP(kJ/kgK)', 'Mass(kg)',
            'FVR', 'CCM', 'TD(°C)', 'CD(°C)', 'HTA(m2)',
            'WTC(W/mK)', 'FTC(W/mK)', 'LPH(L/h)', 'AR', 'IA(°)'
        ]

        # 确保输入特征正确排列
        ordered_features = {feature: input_features[feature] for feature in feature_order}
        input_df = pd.DataFrame([ordered_features])

        # 应用标准化（与训练时相同）
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
            input_df_scaled = pd.DataFrame(input_scaled, columns=feature_order)
            return input_df_scaled
        else:
            return input_df

    except Exception as e:
        st.error(f"Data preprocessing failed: {str(e)}")
        return None


# 主内容区
if uploaded_model is not None and uploaded_scaler is not None:
    # 加载模型和标准化器
    model = load_model(uploaded_model)
    scaler = load_scaler(uploaded_scaler)

    if model is not None and scaler is not None:
        # 特征输入部分
        st.markdown("### 📝 Input parameters")

        # 创建5列，每列3个特征
        col1, col2, col3, col4, col5 = st.columns(5)
        input_features = {}

        # 第一列
        with col1:
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            for feature in list(feature_config.keys())[:3]:
                config = feature_config[feature]
                input_value = st.number_input(
                    f"{feature}",
                    min_value=float(config['min']),
                    max_value=float(config['max']),
                    value=float(config['default']),
                    step=float(config['step']),
                    help=config['description'],
                    key=f"feature_{feature}",
                    format="%.3f"
                )
                input_features[feature] = format_value(input_value)
            st.markdown('</div>', unsafe_allow_html=True)

        # 第二列
        with col2:
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            for feature in list(feature_config.keys())[3:6]:
                config = feature_config[feature]
                input_value = st.number_input(
                    f"{feature}",
                    min_value=float(config['min']),
                    max_value=float(config['max']),
                    value=float(config['default']),
                    step=float(config['step']),
                    help=config['description'],
                    key=f"feature_{feature}",
                    format="%.3f"
                )
                input_features[feature] = format_value(input_value)
            st.markdown('</div>', unsafe_allow_html=True)

        # 第三列
        with col3:
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            for feature in list(feature_config.keys())[6:9]:
                config = feature_config[feature]
                input_value = st.number_input(
                    f"{feature}",
                    min_value=float(config['min']),
                    max_value=float(config['max']),
                    value=float(config['default']),
                    step=float(config['step']),
                    help=config['description'],
                    key=f"feature_{feature}",
                    format="%.3f"
                )
                input_features[feature] = format_value(input_value)
            st.markdown('</div>', unsafe_allow_html=True)

        # 第四列
        with col4:
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            for feature in list(feature_config.keys())[9:12]:
                config = feature_config[feature]
                input_value = st.number_input(
                    f"{feature}",
                    min_value=float(config['min']),
                    max_value=float(config['max']),
                    value=float(config['default']),
                    step=float(config['step']),
                    help=config['description'],
                    key=f"feature_{feature}",
                    format="%.6f"
                )
                input_features[feature] = format_value(input_value)
            st.markdown('</div>', unsafe_allow_html=True)

        # 第五列
        with col5:
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            for feature in list(feature_config.keys())[12:]:
                config = feature_config[feature]
                input_value = st.number_input(
                    f"{feature}",
                    min_value=float(config['min']),
                    max_value=float(config['max']),
                    value=float(config['default']),
                    step=float(config['step']),
                    help=config['description'],
                    key=f"feature_{feature}",
                    format="%.3f"
                )
                input_features[feature] = format_value(input_value)
            st.markdown('</div>', unsafe_allow_html=True)

        # 显示输入特征表格
        st.markdown("### 📋 Input parameter overview")
        formatted_values = [format_value(val) for val in input_features.values()]
        features_display_df = pd.DataFrame({
            'Parameter name': list(input_features.keys()),
            'Parameter value': formatted_values,
            'Parameter description': [feature_config[name]['description'] for name in input_features.keys()]
        })

        pd.options.display.float_format = '{:.6f}'.format
        st.dataframe(features_display_df, use_container_width=True)

        # 预测按钮和结果显示
        st.markdown("---")

        col_pred_left, col_pred_right = st.columns([1, 1])

        with col_pred_left:
            if st.button("🚀 Starting predicting", use_container_width=True):
                with st.spinner("Calculating the predicted value..."):
                    try:
                        # 数据预处理（标准化）
                        processed_data = preprocess_input(input_features, scaler)

                        if processed_data is not None:
                            # 显示预处理信息
                            st.info("✅ Data preprocessing completed ( applied standardisation)")

                            # 进行预测
                            prediction = model.predict(processed_data)[0]

                            # 显示预测结果
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h2>📈 Predicted value</h2>
                                <h1 style="font-size: 3rem; margin: 1rem 0;">{prediction:.3f} W/cm3</h1>
                                <p>According to {len(input_features)} thermodynamic parameters to calculate as</p>
                                <p>Power density predictive value</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # 显示详细预测信息
                            st.info(f"**Predicting power density**: {prediction:.6f} W/kg")

                            # 显示输入参数分布图
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(input_features.keys()),
                                    y=list(input_features.values()),
                                    marker_color='lightblue',
                                    name='Input parameter values'
                                )
                            ])
                            fig.update_layout(
                                title="Input parameter distribution",
                                xaxis_title="Parameter name",
                                yaxis_title="Parameter value",
                                showlegend=True,
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Errors occurred during the predicting process: {str(e)}")

        with col_pred_right:
            st.markdown("### 📊 Parameter distribution visualisation")

            # 创建参数分布饼图
            feature_values = list(input_features.values())
            feature_names = list(input_features.keys())

            normalized_values = [abs(v) / max(abs(v) for v in feature_values) for v in feature_values]

            fig_pie = px.pie(
                values=normalized_values,
                names=feature_names,
                title="Parameter value relative distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # 显示模型信息
            with st.expander("🔍 Model Information"):
                try:
                    if hasattr(model, 'get_params'):
                        params = model.get_params()
                        st.write("Model parameters:")
                        st.json(params)
                except:
                    st.write("Unable to capture model parameter details")

else:
    # 没上传模型时的界面
    st.markdown("""
    ## 👋 Welcome to the power density prediction system for phase-change thermal batteries

    Please follow these steps to start：

    1. **Upload the model file in the left-hand sidebar** (catboost_model.pkl)
    2. **Upload scaler data files** (Scaler.pkl)
    3. **Input feature parameters** - Fifteen parameters can be input
    4. **Click the predict button** - Obtain power density prediction results

    ### 📋 System parameters description：

    - **LH(kJ/kg)**: Latent heat
    - **MT(°C)**: Melt temperature
    - **TC(W/m2K)**: Thermal conductivity
    - **CP(kJ/kgK)**: Specific heat capacity
    - **Mass(kg)**: Mass
    - **FVR**: Fin volume ratio
    - **CCM**: Close-contact melting
    - **TD(°C)**: Thermal temperature difference
    - **CD(°C)**: Cold temperature difference
    - **HTA(m2)**: Heat transfer area
    - **WTC(W/m2K)**: Wall thermal conductivity
    - **FTC(W/m2K)**: Fluid thermal conductivity
    - **LPH(L/h)**: litres per hour
    - **AR**: Aspect ratio
    - **IA(°)**: Inclination angle

    ### 💡 Usage tips：

    - Input values are rounded to six decimals
    - Automatically apply the same data scaler as in training
    - Predicted result is power density (W/cm3)
    - Hovering the mouse over a parameter name will display its description
    """)

# 页脚信息
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Power density prediction system for phase-change thermal batteries | CatBoost regression model | Building with Streamlit"
    "</div>",
    unsafe_allow_html=True
)