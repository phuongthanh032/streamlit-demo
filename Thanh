import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import streamlit as st

st.set_page_config(page_title="Xử lý tín hiệu y sinh", layout="wide")

st.title("Ứng dụng Demo Xử lý Tín hiệu Y sinh")
st.caption("Minh họa tạo/phân tích tín hiệu, lọc nhiễu và phổ tần số trên Streamlit")

# Thanh điều khiển bên trái
st.sidebar.header("Tham số tín hiệu")
signal_type = st.sidebar.selectbox(
    "Loại tín hiệu",
    ["ECG (Điện tâm đồ - mô phỏng)", "EOG (Điện nhãn đồ - mô phỏng)", "PCG (Âm tim - mô phỏng)"]
)
fs = st.sidebar.slider("Tần số lấy mẫu fs (Hz)", min_value=100, max_value=1000, value=250, step=50)
duration = st.sidebar.slider("Thời lượng (giây)", min_value=2.0, max_value=10.0, value=5.0, step=0.5)
noise_level = st.sidebar.slider("Mức nhiễu trắng", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
powerline_noise = st.sidebar.checkbox("Thêm nhiễu đường dây 50 Hz", value=True)

st.sidebar.header("Bộ lọc thông thấp (Low-pass)")
use_filter = st.sidebar.checkbox("Áp dụng bộ lọc Butterworth", value=True)
cutoff = st.sidebar.slider("Tần số cắt (Hz)", min_value=5.0, max_value=float(fs // 2 - 1), value=35.0, step=1.0)

# Tạo trục thời gian và tín hiệu mẫu
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

if signal_type.startswith("ECG"):
    # Mô phỏng nhịp tim ~72 bpm (1.2 Hz) bằng tổng các xung Gaussian
    hr_hz = 1.2
    beat_phase = (t * hr_hz) % 1.0
    clean_sig = (
        1.0 * np.exp(-((beat_phase - 0.25) / 0.015) ** 2)   # Đỉnh R
        - 0.2 * np.exp(-((beat_phase - 0.22) / 0.012) ** 2)  # Sóng Q
        - 0.25 * np.exp(-((beat_phase - 0.28) / 0.012) ** 2) # Sóng S
        + 0.2 * np.exp(-((beat_phase - 0.16) / 0.025) ** 2)  # Sóng P
        + 0.3 * np.exp(-((beat_phase - 0.45) / 0.04) ** 2)   # Sóng T
    )
elif signal_type.startswith("EOG"):
    # Mô phỏng chuyển động liếc mắt bậc thang
    clean_sig = np.zeros_like(t)
    step_times = [1.0, 2.2, 3.4, 4.3]
    levels = [0.5, -0.4, 0.7, 0.0]
    for idx, st_t in enumerate(step_times):
        clean_sig += levels[idx] * 0.5 * (1 + np.tanh((t - st_t) / 0.08))
else:
    # Mô phỏng âm tim S1 - S2
    hr_hz = 1.1
    phase = (t * hr_hz) % 1.0
    s1 = np.sin(2 * np.pi * 45 * t) * np.exp(-((phase - 0.15) / 0.03) ** 2)
    s2 = 0.7 * np.sin(2 * np.pi * 60 * t) * np.exp(-((phase - 0.42) / 0.025) ** 2)
    clean_sig = s1 + s2

rng = np.random.default_rng(42)
noisy_sig = clean_sig + noise_level * rng.standard_normal(len(t))
if powerline_noise:
    noisy_sig += 0.25 * np.sin(2 * np.pi * 50 * t)

if use_filter:
    b, a = butter(4, cutoff / (0.5 * fs), btype="low")
    processed_sig = filtfilt(b, a, noisy_sig)
else:
    processed_sig = noisy_sig.copy()

# Thống kê nhanh
col1, col2, col3 = st.columns(3)
col1.metric("Số mẫu tín hiệu", f"{len(t)}")
col2.metric("Biên độ cực đại (sau xử lý)", f"{np.max(processed_sig):.2f}")
col3.metric("Độ lệch chuẩn (Std)", f"{np.std(processed_sig):.3f}")

# Vẽ đồ thị miền thời gian
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, noisy_sig, label="Tín hiệu gốc (+ nhiễu)", alpha=0.55, linewidth=1)
ax.plot(t, processed_sig, label="Tín hiệu sau xử lý", linewidth=1.8)
ax.set_xlabel("Thời gian (s)")
ax.set_ylabel("Biên độ (đv tương đối)")
ax.set_title(f"Miền thời gian - {signal_type}")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)
st.pyplot(fig)

# Phổ tần số FFT
freqs = np.fft.rfftfreq(len(processed_sig), d=1.0 / fs)
fft_mag = np.abs(np.fft.rfft(processed_sig)) / len(processed_sig)

fig_fft, ax_fft = plt.subplots(figsize=(10, 3))
ax_fft.plot(freqs, fft_mag, linewidth=1.3)
ax_fft.set_xlim(0, min(100, fs / 2))
ax_fft.set_xlabel("Tần số (Hz)")
ax_fft.set_ylabel("Biên độ phổ")
ax_fft.set_title("Phổ biên độ (FFT)")
ax_fft.grid(True, linestyle="--", alpha=0.4)
st.pyplot(fig_fft)

# Cho phép tải dữ liệu đã xử lý dạng CSV
df_out = pd.DataFrame({
    "time_s": t,
    "raw_signal": noisy_sig,
    "processed_signal": processed_sig
})
st.download_button(
    label="Tải dữ liệu đã xử lý (.csv)",
    data=df_out.to_csv(index=False).encode("utf-8"),
    file_name="processed_biosignal.csv",
    mime="text/csv"
)
