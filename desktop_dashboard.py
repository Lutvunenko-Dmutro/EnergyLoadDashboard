import tkinter as tk
import customtkinter as ctk  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

# --- Налаштування стилю Matplotlib для темної теми ---
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "#2B2B2B",
    "figure.edgecolor": "#2B2B2B",
    "axes.facecolor": "#3C3C3C",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#555555",
    "legend.facecolor": "#3C3C3C",
    "legend.edgecolor": "none",
    "legend.labelcolor": "white"
})

# --- 1. Функції джерел даних ---

def generate_sample_data(days=60):
    print("--- Генерація зразкових даних (60 днів) ---")
    periods = days * 24
    time_index = pd.date_range(start='2025-01-01', periods=periods, freq='h')
    base_load = 500; trend = np.linspace(0, 50, num=periods)
    time_of_day = time_index.hour
    daily_seasonality = 100 * (np.sin((time_of_day - 3) * (2*np.pi/24)) + 
                             np.sin((time_of_day - 8) * (2*np.pi/24)))
    day_of_week = time_index.dayofweek
    weekly_seasonality = -70 * ((day_of_week == 5) | (day_of_week == 6)).astype(int)
    noise = np.random.normal(0, 15, periods)
    load_mw = base_load + trend + daily_seasonality + weekly_seasonality + noise
    ts_data = pd.Series(load_mw, index=time_index, name="навантаження_мвт")
    print("Зразкові дані успішно згенеровано.")
    return ts_data

def load_csv_file():
    """
    Намагається завантажити CSV. Повертає дані або None, якщо помилка.
    """
    try:
        df = pd.read_csv("power_load_hourly.csv", 
                         parse_dates=['мітка_часу'], 
                         index_col='мітка_часу')
        
        ts_data = df['навантаження_мвт'].asfreq('h').ffill() 
        
        print("Дані 'power_load_hourly.csv' успішно завантажено.")
        return ts_data
    except FileNotFoundError:
        print("ПОМИЛКА: Файл 'power_load_hourly.csv' не знайдено.")
        return None
    except Exception as e:
        print(f"ПОМИЛKA ЧИТАННЯ ФАЙЛУ: {e}")
        return None

# --- 2. Функції для створення вкладок ---

# Q1: "Особливості/відмінності"
def create_tab1_features(container, data):
    label = ctk.CTkLabel(container, text="Q1: Динамічний ряд та його Особливості",
                         font=ctk.CTkFont(size=16, weight="bold"))
    label.pack(pady=10, padx=10, fill='x')

    fig1 = Figure(figsize=(10, 3.5))
    ax1 = fig1.add_subplot(111)
    data.plot(ax=ax1, title="1. Динамічний часовий ряд (Ваші дані)")
    ax1.set_ylabel("Навантаження (МВт)"); ax1.grid(True)
    fig1.tight_layout()
    canvas1 = FigureCanvasTkAgg(fig1, master=container)
    canvas1.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
    
    fig2 = Figure(figsize=(10, 3.5))
    ax2 = fig2.add_subplot(111)
    plot_acf(data.dropna(), lags=48, ax=ax2, title="2. Особливість: Автокореляція (Пам'ять ряду)")
    ax2.grid(True)
    fig2.tight_layout()
    canvas2 = FigureCanvasTkAgg(fig2, master=container)
    canvas2.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)

    text = ("Особливість 1 (Графік 1): Залежність від часу. Порядок даних критичний.\n"
            "Особливість 2 (Графік 2): Висока автокореляція. Значення lag=24 (24 год тому)\n"
            "майже так само сильно впливає, як lag=1 (1 год тому). Це доводить сезонність.")
    label_info = ctk.CTkLabel(container, text=text, justify="left", font=ctk.CTkFont(size=12))
    label_info.pack(fill='x', padx=10, pady=(0, 10))


# Q2: "Як... подані?"
def create_tab2_components(container, data):
    data_subset = data.head(24 * 30).dropna() 

    # Фрейм для кнопок
    control_frame = ctk.CTkFrame(container, fg_color="transparent")
    control_frame.pack(fill='x', padx=10, pady=(5,0))
    
    label = ctk.CTkLabel(control_frame, text="Q2: 'Як подані компоненти?' ->", 
                         font=ctk.CTkFont(size=16, weight="bold"))
    label.pack(side='left', padx=(0, 10))

    # Фрейм для графіків
    plot_frame = ctk.CTkFrame(container, fg_color="transparent")
    plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill='both', expand=True)

    def update_decomposition():
        model_type = model_var.get()
        if len(data_subset) < 2 * 24:
             for ax in axes: ax.clear()
             axes[0].text(0.5, 0.5, 'Помилка: недостатньо даних для декомпозиції\n(потрібно мінімум 48 годин)', 
                          ha='center', va='center', color='red', transform=axes[0].transAxes)
             canvas.draw(); return
             
        decomposition = seasonal_decompose(data_subset, model=model_type, period=24)
        for ax in axes: ax.clear()
            
        decomposition.observed.plot(ax=axes[0], legend=False); axes[0].set_ylabel('Original')
        decomposition.trend.plot(ax=axes[1], legend=False); axes[1].set_ylabel('Trend')
        decomposition.seasonal.plot(ax=axes[2], legend=False); axes[2].set_ylabel('Seasonal')
        decomposition.resid.plot(ax=axes[3], legend=False); axes[3].set_ylabel('Residual')
        
        fig.suptitle(f"Q2: Компоненти (Модель: {model_type.capitalize()})", y=1.02)
        fig.tight_layout()
        canvas.draw()

    model_var = tk.StringVar(value='additive') 
    radio_add = ctk.CTkRadioButton(control_frame, text="Адитивна (Y = T+S+E)", 
                                   variable=model_var, value='additive', command=update_decomposition)
    radio_add.pack(side='left', padx=5)
    radio_mul = ctk.CTkRadioButton(control_frame, text="Мультиплікативна (Y = T*S*E)", 
                                   variable=model_var, value='multiplicative', command=update_decomposition)
    radio_mul.pack(side='left', padx=5)
    
    update_decomposition()


# Q3: "Як... визначити?"
def create_tab3_trend(container, data_in):
    label = ctk.CTkLabel(container, text="Q3: 'Як визначити тренд?' -> Метод: Візуальне згладжування (SMA)",
                         font=ctk.CTkFont(size=16, weight="bold"))
    label.pack(pady=10, padx=10, fill='x')
    
    trend_sma = data_in.rolling(window=24 * 30, center=True).mean()
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    data_in.plot(ax=ax, alpha=0.3, label='Оригінал')
    trend_sma.plot(ax=ax, color='red', linewidth=3, label='Тренд: SMA (30 днів)')
    
    ax.set_title("Питання 3: Визначення тренду")
    ax.set_ylabel("Навантаження (МВт)"); ax.legend(); ax.grid(True)
    fig.tight_layout()

    text = ("Для визначення тренду використовується метод ковзного середнього (SMA) за 30 днів.\n"
            "Червона лінія показує довгострокову тенденцію зміни навантаження, згладжуючи короткострокові коливання.")
    label_info = ctk.CTkLabel(container, text=text, justify="left", font=ctk.CTkFont(size=12))
    label_info.pack(pady=5, padx=10, fill='x')
    
    canvas = FigureCanvasTkAgg(fig, master=container)
    canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=(0, 10))
    canvas.draw()


# Q4: "Які способи..."
def create_tab4_smoothing(container, data):
    ts_sample = data.head(24 * 7).dropna() 

    # Фрейм для слайдера
    control_frame = ctk.CTkFrame(container, fg_color="transparent")
    control_frame.pack(fill='x', padx=10, pady=5)
    
    label = ctk.CTkLabel(control_frame, text="Q4: 'Які способи?' -> Демонстрація SMA", 
                         font=ctk.CTkFont(size=16, weight="bold"))
    label.pack(side='left', padx=5)
    
    slider_label = ctk.CTkLabel(control_frame, text="Вікно згладжування: 2 год", 
                                font=ctk.CTkFont(size=12))
    slider_label.pack(side='right', padx=10)

    # Фрейм для графіку
    plot_frame = ctk.CTkFrame(container, fg_color="transparent")
    plot_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def update_smoothing(slider_value):
        window_size = int(float(slider_value))
        if window_size > len(ts_sample): window_size = len(ts_sample)
        if window_size < 2: window_size = 2
            
        sma = ts_sample.rolling(window=window_size, center=True).mean()
        
        ax.clear()
        ts_sample.plot(ax=ax, style='.', alpha=0.5, label='Оригінал (7 днів)')
        sma.plot(ax=ax, label=f'SMA (Вікно = {window_size} год)', color='green', linewidth=3)
        ax.set_title("Питання 4: Інтерактивне згладжування (SMA)")
        ax.legend(); ax.grid(True); fig.tight_layout()
        canvas.draw()
        slider_label.configure(text=f"Вікно згладжування: {window_size} год")

    slider = ctk.CTkSlider(control_frame, from_=2, to=72, 
                           command=update_smoothing)
    slider.set(2); slider.pack(fill='x', expand=True, padx=10, side='left')
    
    if not ts_sample.empty: update_smoothing(2)
    else:
        ax.text(0.5, 0.5, 'Помилка: недостатньо даних для згладжування', 
                 ha='center', va='center', color='red'); canvas.draw()

# Q5: "STATISTICA"
def create_tab5_statistica(container):
    text_content = """
Q5: Для чого STATISTICA та її модулі у Вашому проекті?
========================================================

STATISTICA — це інтегрована платформа, що реалізує
повний цикл аналізу. У контексті мого проекту
"Система моніторингу завантаженості енергосистеми",
ключові модулі STATISTICA були б використані так:

1. Модуль: Basic Statistics
---------------------------------
   Завдання: Первинний аналіз та очищення даних.
   
   У проекті: Використовувався б для знаходження 
   статистичних характеристик мого ряду (навантаження_мвт):
   середнє, медіана, стандартне відхилення. Допоміг би 
   ідентифікувати аномальні викиди (наприклад, збої лічильників
   або аварії) для їх корекції.

2. Модуль: Graphs
---------------------------------
   Завдання: Візуалізація.
   
   У проекті: Побудова графіків, аналогічних тим, що 
   представлені на Вкладках 1-4 цього дашборду.

3. Модуль: Time Series Analysis & Forecasting (Найважливіший)
---------------------------------
   Завдання: Аналіз структури ряду та прогноз.
   
   У проекті: Це ядро моєї системи моніторингу.
     - Декомпозиція (як на Вкладці 2): Для виділення 
       чистого тренду та сезонного профілю споживання.
     - Прогнозування (ARIMA / Holt-Winters): Побудова 
       моделі для прогнозування навантаження на 24/48 
       годин вперед. Це ключова функція моніторингу 
       для запобігання перевантаженням.

4. Модуль: Data Miner / Machine Learning
---------------------------------
   Завдання: Побудова складних нелінійних моделей.
   
   У проекті:
     - Регресія (напр., Neural Networks): Побудова 
       прогнозу, який враховує не тільки час, але й 
       зовнішні фактори з мого CSV (температура,
       швидкість вітру, свято).
     - Кластеризація: Автоматичне групування днів 
       за профілем навантаження (напр., "Робочий день",
       "Вихідний", "Аномальна спека"). Це дозволяє 
       системі моніторингу мати різні сценарії.
"""
    textbox = ctk.CTkTextbox(container, wrap="word", 
                             font=ctk.CTkFont(family="Arial", size=13),
                             fg_color="#333333")
    textbox.pack(fill='both', expand=True, padx=10, pady=10)
    
    textbox.insert("1.0", text_content)
    textbox.configure(state='disabled') # Тільки для читання

# --- 3. Головна функція для запуску програми ---
def main():
    
    ctk.set_appearance_mode("dark")  
    ctk.set_default_color_theme("blue") 

    root = ctk.CTk() # Головне вікно
    root.title("Сучасний Дашборд 'Система Моніторингу' (v6, CustomTkinter)")
    root.geometry("1100x900")

    # --- ФРЕЙМ ДЛЯ КЕРУВАННЯ ---
    control_frame = ctk.CTkFrame(root)
    control_frame.pack(fill='x', pady=10, padx=10)

    status_label = ctk.CTkLabel(control_frame, text="Статус: Оберіть джерело даних", 
                                font=ctk.CTkFont(size=12, slant="italic"))
    status_label.pack(side='right', padx=10)

    # --- Головний "Нотатник" -> CTkTabview ---
    tab_control = ctk.CTkTabview(root, width=1050)
    tab_control.pack(expand=1, fill='both', padx=10, pady=(0,10))
    
    # Додаємо вкладки
    tab1_container = tab_control.add('Q1: Особливості')
    tab2_container = tab_control.add('Q2: Компоненти')
    tab3_container = tab_control.add('Q3: Тренд')
    tab4_container = tab_control.add('Q4: Згладжування')
    tab5_container = tab_control.add('Q5: STATISTICA')
    
    # --- ЛОГІКА: Функція перебудови ---
    all_containers = [tab1_container, tab2_container, tab3_container, 
                    tab4_container, tab5_container]

    def build_all_tabs(data):
        for container in all_containers:
            for widget in container.winfo_children():
                widget.destroy()
        
        print("Будуємо/перебудовуємо вкладки...")
        create_tab1_features(tab1_container, data)
        create_tab2_components(tab2_container, data)
        create_tab3_trend(tab3_container, data) 
        create_tab4_smoothing(tab4_container, data)
        create_tab5_statistica(tab5_container) 
        print("Вкладки збудовано.")
    
    # --- ФУНКЦІЇ ДЛЯ КНОПОК ---
    def on_load_csv_click():
        data = load_csv_file()
        if data is not None:
            build_all_tabs(data)
            status_label.configure(text="Статус: Завантажено 'power_load_hourly.csv'", 
                                   text_color='green')
        else:
            status_label.configure(text="ПОМИЛКА: Файл 'power_load_hourly.csv' не знайдено.", 
                                   text_color='red')
            for container in all_containers:
                for widget in container.winfo_children():
                    widget.destroy()
            create_tab5_statistica(tab5_container)

    def on_generate_click():
        data = generate_sample_data()
        build_all_tabs(data)
        status_label.configure(text="Статус: Використовуються згенеровані дані", 
                               text_color='cyan')

    # --- Додаємо кнопки у фрейм керування ---
    load_csv_button = ctk.CTkButton(control_frame, text="Завантажити CSV", 
                                    command=on_load_csv_click)
    load_csv_button.pack(side='left', padx=10)

    load_gen_button = ctk.CTkButton(control_frame, text="Згенерувати дані (Приклад)", 
                                    command=on_generate_click, fg_color="#4A4A4A", 
                                    hover_color="#555555")
    load_gen_button.pack(side='left', padx=10)

    # --- Початковий запуск ---
    on_load_csv_click()

    # --- Запуск головного циклу ---
    print("Запуск десктопного дашборду (CustomTkinter)...")
    root.mainloop()
    print("Дашборд закрито.")

# --- Точка входу ---
if __name__ == "__main__":
    main()