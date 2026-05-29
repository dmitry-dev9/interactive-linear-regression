import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# 1. Настройка страницы
st.set_page_config(page_title="Регрессия", layout="wide") # Широкий экран для красивых графиков
st.title("📊 Интерактивная линейная регрессия")

# Инициализация seed
np.random.seed(42)

# 2. Боковая панель для управления (так интерфейс выглядит чище)
with st.sidebar:
    st.header("⚙️ Настройки данных")
    n = st.slider("Количество точек:", 10, 200, 50)
    noise = st.slider("Уровень шума:", 0.0, 5.0, 1.0)
    
    TREND_FUNCTIONS = {
        "Линейный рост": lambda x, n: 2 * x + n,
        "Линейный спад": lambda x, n: -2 * x + n,
        "Квадратичный вверх": lambda x, n: x**2 + n,
        "Квадратичный вниз": lambda x, n: -(x**2) + n,
    }
    trend = st.selectbox("Тип тренда", list(TREND_FUNCTIONS.keys()))

# 3. Генерация и кэширование данных
params = (n, noise, trend)
if "params" not in st.session_state or st.session_state["params"] != params:
    x = np.linspace(0, 10, n)
    noise_term = np.random.normal(0, noise, n)
    y = TREND_FUNCTIONS[trend](x, noise_term)
    st.session_state["data"] = (x, y)
    st.session_state["params"] = params

x, y = st.session_state["data"]

# 4. Математика: Линейная регрессия (МНК)
X = np.c_[np.ones(n), x]
w = np.linalg.inv(X.T @ X) @ X.T @ y
y_pred = X @ w
residuals = y - y_pred

# Метрики
mse = np.mean(residuals ** 2)
r2 = 1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)

# 5. Интерфейс: Главный график и Метрики
st.subheader("🤖 Модель и метрики эффективности")
show_line = st.checkbox("Показать линию регрессии", value=True)

# Выводим метрики в красивых карточках
col_m1, col_m2 = st.columns(2)
col_m1.metric("Ошибка (MSE)", round(mse, 3))
col_m2.metric("Качество ($R^2$)", round(r2, 3))

# Рисуем главный график
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(x, y, color="gray", alpha=0.7, label="Реальные данные")
if show_line:
    idx = np.argsort(x)
    ax.plot(x[idx], y_pred[idx], color="red", lw=2, label="Линия регрессии (МНК)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# 6. Интерфейс: Анализ остатков
st.markdown("---")
st.subheader("📉 Диагностика модели (Анализ остатков)")

# Используем колонки Streamlit, чтобы графики не сжимались в кашу
col1, col2, col3 = st.columns(3)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_pred, residuals, alpha=0.6, color="purple")
    ax1.axhline(0, color="black", linestyle="--")
    ax1.set_title("Остатки vs Предсказания")
    ax1.set_xlabel("Предсказанные значения")
    ax1.set_ylabel("Остатки")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.histplot(residuals, kde=True, color="teal", ax=ax2)
    ax2.set_title("Распределение остатков")
    ax2.set_xlabel("Значение остатка")
    st.pyplot(fig2)

with col3:
    fig3, ax3 = plt.subplots()
    stats.probplot(residuals, plot=ax3)
    ax3.set_title("Q-Q Plot (Нормальность)")
    st.pyplot(fig3)
# привет