"""Property Price Predictor — interactive Tkinter desktop application.

Tabs:
    1. Overview    — dataset snapshot, missing-value chart, target distribution.
    2. Preprocess  — configurable cleaning / encoding / PCA with live diagnostics.
    3. Train       — RandomForest hyper-parameters, threaded training, R2/RMSE/MAE.
    4. Predict     — interactive form producing a live price estimate.
    5. Insights    — exploratory plots (correlation, region, type, rooms, year).

Run:
    python app.py
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import pandas as pd

from preprocessing import TARGET, DataPreprocessor, load_data
from model import ModelTrainer


APP_TITLE = "Property Price Predictor — Melbourne"
DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Property Sales of Melbourne City.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_predictor.joblib")

plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#181825",
    "axes.edgecolor": "#cdd6f4",
    "axes.labelcolor": "#cdd6f4",
    "xtick.color": "#cdd6f4",
    "ytick.color": "#cdd6f4",
    "text.color": "#cdd6f4",
    "axes.grid": True,
    "grid.color": "#313244",
    "grid.alpha": 0.6,
})

ACCENT = "#89b4fa"
ACCENT2 = "#f5c2e7"
SUCCESS = "#a6e3a1"
WARN = "#f9e2af"
DANGER = "#f38ba8"
BG = "#1e1e2e"
BG2 = "#181825"
PANEL = "#313244"
FG = "#cdd6f4"
MUTED = "#a6adc8"


def fmt_money(v: float) -> str:
    return f"A$ {v:,.0f}"


class ScrollFrame(ttk.Frame):
    """A vertically scrollable container."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=BG2)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_configure(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.window_id, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class PlotFrame(ttk.Frame):
    """A panel that hosts a matplotlib figure + toolbar."""

    def __init__(self, parent, figsize=(7, 4.5), *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.fig, self.ax = plt.subplots(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

    def clear(self):
        self.ax.clear()

    def draw(self):
        self.canvas.draw()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x820")
        self.minsize(1080, 700)
        self.configure(bg=BG)

        self.df: pd.DataFrame | None = None
        self.preprocessor: DataPreprocessor | None = None
        self.trainer: ModelTrainer | None = None
        self.train_queue: queue.Queue = queue.Queue()
        self._train_thread: threading.Thread | None = None

        self._setup_style()
        self._build_layout()
        self._bind_shortcuts()

        if os.path.exists(DEFAULT_CSV):
            self.after(100, lambda: self._load_csv(DEFAULT_CSV))

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Card.TFrame", background=BG2)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=BG2, foreground=FG)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("Title.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI Semibold", 18))
        style.configure("H2.TLabel", background=BG, foreground=FG, font=("Segoe UI Semibold", 13))
        style.configure("H3.TLabel", background=BG2, foreground=ACCENT, font=("Segoe UI Semibold", 11))
        style.configure("Stat.TLabel", background=BG2, foreground=ACCENT, font=("Segoe UI Semibold", 16))
        style.configure("Big.TLabel", background=BG2, foreground=SUCCESS, font=("Segoe UI Semibold", 22))
        style.configure("Status.TLabel", background=BG2, foreground=MUTED, font=("Segoe UI", 9))

        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG2, foreground=MUTED,
                         padding=(18, 8), font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", PANEL)], foreground=[("selected", ACCENT)])

        style.configure("TButton", background=PANEL, foreground=FG,
                         borderwidth=0, padding=(14, 7), font=("Segoe UI", 10))
        style.map("TButton",
                  background=[("active", ACCENT), ("disabled", BG2)],
                  foreground=[("active", BG), ("disabled", MUTED)])
        style.configure("Accent.TButton", background=ACCENT, foreground=BG,
                         padding=(16, 8), font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton",
                  background=[("active", "#74c7ec")], foreground=[("active", BG)])
        style.configure("Success.TButton", background=SUCCESS, foreground=BG,
                         padding=(14, 7), font=("Segoe UI Semibold", 10))
        style.map("Success.TButton", background=[("active", "#94e2d5")])

        style.configure("TEntry", fieldbackground=BG2, foreground=FG, insertcolor=FG, borderwidth=1)
        style.configure("TCombobox", fieldbackground=BG2, foreground=FG, arrowcolor=ACCENT)
        style.map("TCombobox", fieldbackground=[("readonly", BG2)], foreground=[("readonly", FG)])
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("TRadiobutton", background=BG, foreground=FG)
        style.map("TRadiobutton", background=[("active", BG)])
        style.configure("Horizontal.TScale", background=BG)
        style.configure("TSpinbox", fieldbackground=BG2, foreground=FG, arrowcolor=ACCENT)
        style.configure("TProgressbar", background=ACCENT, troughcolor=BG2, borderwidth=0)
        style.configure("TSeparator", background=PANEL)
        style.configure("TLabelframe", background=BG2, foreground=ACCENT, bordercolor=PANEL)
        style.configure("TLabelframe.Label", background=BG2, foreground=ACCENT)

    def _build_layout(self):
        header = ttk.Frame(self)
        header.pack(fill="x", padx=16, pady=(12, 4))
        ttk.Label(header, text="🏠  Property Price Predictor", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Data preprocessing · PCA · Random Forest · R² evaluation",
                  style="Muted.TLabel").pack(side="left", padx=16, pady=(6, 0))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").pack(side="right", pady=(8, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=16, pady=8)

        self.tab_overview = ttk.Frame(self.notebook)
        self.tab_preprocess = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_predict = ttk.Frame(self.notebook)
        self.tab_insights = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_overview, text="① Overview")
        self.notebook.add(self.tab_preprocess, text="② Preprocess")
        self.notebook.add(self.tab_train, text="③ Train")
        self.notebook.add(self.tab_predict, text="④ Predict")
        self.notebook.add(self.tab_insights, text="⑤ Insights")

        self._build_overview()
        self._build_preprocess()
        self._build_train()
        self._build_predict()
        self._build_insights()

    def _bind_shortcuts(self):
        self.bind("<Control-o>", lambda e: self._browse_csv())
        self.bind("<Control-s>", lambda e: self._save_model())

    def set_status(self, text: str):
        self.status_var.set(text)

    def _card(self, parent, text=None):
        card = ttk.Frame(parent, style="Card.TFrame")
        if text:
            ttk.Label(card, text=text, style="H3.TLabel").pack(anchor="w", padx=12, pady=(10, 4))
        return card

    # ------------------------------------------------------------------ Overview
    def _build_overview(self):
        root = ScrollFrame(self.tab_overview)
        root.pack(fill="both", expand=True)
        body = root.inner

        bar = ttk.Frame(body)
        bar.pack(fill="x", padx=12, pady=10)
        ttk.Button(bar, text="Open CSV…", command=self._browse_csv).pack(side="left")
        self.csv_label = ttk.Label(bar, text="No file loaded.", style="Muted.TLabel")
        self.csv_label.pack(side="left", padx=12)

        stats = ttk.Frame(body)
        stats.pack(fill="x", padx=12, pady=(0, 10))
        self.stat_rows_label = ttk.Label(stats, text="", style="Card.TLabel")
        self.stat_cols_label = ttk.Label(stats, text="", style="Card.TLabel")
        self.stat_target_label = ttk.Label(stats, text="", style="Card.TLabel")
        self.stat_missing_label = ttk.Label(stats, text="", style="Card.TLabel")
        for i, w in enumerate([self.stat_rows_label, self.stat_cols_label, self.stat_target_label, self.stat_missing_label]):
            card = ttk.Frame(stats, style="Card.TFrame")
            card.grid(row=0, column=i, padx=6, sticky="nsew")
            stats.grid_columnconfigure(i, weight=1)
            w.pack(in_=card, padx=14, pady=12)

        preview_card = self._card(body, "Data preview (first 5 rows)")
        preview_card.pack(fill="x", padx=12, pady=6)
        self.preview_text = tk.Text(preview_card, height=8, bg=BG2, fg=FG, insertbackground=FG,
                                    relief="flat", font=("Consolas", 9), wrap="none")
        self.preview_text.pack(fill="x", padx=12, pady=(0, 10))

        plots = ttk.Frame(body)
        plots.pack(fill="both", expand=True, padx=12, pady=6)
        self.overview_missing = PlotFrame(plots, figsize=(7, 3.6))
        self.overview_missing.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.overview_target = PlotFrame(plots, figsize=(7, 3.6))
        self.overview_target.pack(side="left", fill="both", expand=True, padx=(6, 0))

    def _browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select property sales CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.path.dirname(DEFAULT_CSV),
        )
        if path:
            self._load_csv(path)

    def _load_csv(self, path: str):
        self.set_status("Loading data…")
        try:
            self.df = load_data(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            self.set_status("Load failed.")
            return

        self.csv_label.config(text=os.path.basename(path))
        n_rows, n_cols = self.df.shape
        self.stat_rows_label.config(text=f"Rows\n{n_rows:,}", style="Stat.TLabel")
        self.stat_cols_label.config(text=f"Columns\n{n_cols}", style="Stat.TLabel")
        if TARGET in self.df.columns:
            self.stat_target_label.config(text=f"Target\n{TARGET}", style="Stat.TLabel")
        missing_pct = self.df.isnull().mean().mean() * 100
        self.stat_missing_label.config(text=f"Avg missing\n{missing_pct:.1f}%", style="Stat.TLabel")

        preview = self.df.head(5).to_string()
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", preview)

        self._plot_missing()
        self._plot_target_distribution()
        self._populate_predict_form()
        self._refresh_insights()
        self.set_status(f"Loaded {n_rows:,} rows × {n_cols} columns.")

    def _plot_missing(self):
        ax = self.overview_missing.ax
        ax.clear()
        if self.df is None:
            return
        miss = self.df.isnull().sum()
        miss = miss[miss > 0].sort_values()
        if miss.empty:
            ax.text(0.5, 0.5, "No missing values 🎉", ha="center", va="center", transform=ax.transAxes, fontsize=13)
            ax.set_axis_off()
        else:
            ax.barh(miss.index, miss.values, color=ACCENT)
            ax.set_title("Missing values per column", color=ACCENT)
            ax.tick_params(axis="y", labelsize=8)
        self.overview_missing.draw()

    def _plot_target_distribution(self):
        ax = self.overview_target.ax
        ax.clear()
        if self.df is None or TARGET not in self.df.columns:
            return
        ax.hist(self.df[TARGET], bins=50, color=ACCENT2, edgecolor=BG)
        ax.set_title(f"{TARGET} distribution", color=ACCENT)
        ax.ticklabel_format(style="plain", axis="x")
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(20)
            lbl.set_fontsize(8)
        self.overview_target.draw()

    # --------------------------------------------------------------- Preprocess
    def _build_preprocess(self):
        root = ScrollFrame(self.tab_preprocess)
        root.pack(fill="both", expand=True)
        body = root.inner

        ctrl = ttk.Frame(body)
        ctrl.pack(fill="x", padx=12, pady=10)

        left = ttk.LabelFrame(ctrl, text="Cleaning")
        left.pack(side="left", fill="y", padx=(0, 8))
        self.drop_address_var = tk.BooleanVar(value=True)
        self.drop_index_var = tk.BooleanVar(value=True)
        self.cap_outliers_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Drop Address column", variable=self.drop_address_var).pack(anchor="w", padx=8, pady=4)
        ttk.Checkbutton(left, text="Drop index column", variable=self.drop_index_var).pack(anchor="w", padx=8, pady=4)
        ttk.Checkbutton(left, text="Cap outliers (IQR)", variable=self.cap_outliers_var).pack(anchor="w", padx=8, pady=4)

        mid = ttk.LabelFrame(ctrl, text="Imputation")
        mid.pack(side="left", fill="y", padx=8)
        self.num_impute_var = tk.StringVar(value="median")
        self.cat_impute_var = tk.StringVar(value="most_frequent")
        for val, label in [("median", "Numeric: median"), ("mean", "Numeric: mean")]:
            ttk.Radiobutton(mid, text=label, variable=self.num_impute_var, value=val).pack(anchor="w", padx=8, pady=4)
        for val, label in [("most_frequent", "Categorical: mode"), ("constant", "Categorical: 'unknown'")]:
            ttk.Radiobutton(mid, text=label, variable=self.cat_impute_var, value=val).pack(anchor="w", padx=8, pady=4)

        right = ttk.LabelFrame(ctrl, text="Scaling & PCA")
        right.pack(side="left", fill="y", padx=8)
        self.scale_var = tk.BooleanVar(value=True)
        self.pca_var = tk.BooleanVar(value=True)
        self.pca_mode_var = tk.StringVar(value="variance")
        self.pca_var_entry = ttk.Spinbox(right, from_=2, to=200, increment=1, width=6)
        self.pca_var_entry.set(0.95)
        ttk.Checkbutton(right, text="Standardise features", variable=self.scale_var).pack(anchor="w", padx=8, pady=4)
        ttk.Checkbutton(right, text="Apply PCA", variable=self.pca_var).pack(anchor="w", padx=8, pady=4)
        mode_row = ttk.Frame(right)
        mode_row.pack(fill="x", padx=8, pady=4)
        ttk.Radiobutton(mode_row, text="Variance", variable=self.pca_mode_var, value="variance").pack(side="left")
        ttk.Radiobutton(mode_row, text="Components", variable=self.pca_mode_var, value="components").pack(side="left", padx=8)
        ttk.Label(right, text="Value (0-1 variance or int components):").pack(anchor="w", padx=8)
        self.pca_var_entry.pack(anchor="w", padx=8, pady=(0, 8))

        actions = ttk.Frame(body)
        actions.pack(fill="x", padx=12, pady=(0, 8))
        ttk.Button(actions, text="Run preprocessing", style="Accent.TButton",
                   command=self._run_preprocess).pack(side="left")
        self.preprocess_status = ttk.Label(actions, text="", style="Muted.TLabel")
        self.preprocess_status.pack(side="left", padx=12)

        plots = ttk.Frame(body)
        plots.pack(fill="both", expand=True, padx=12, pady=6)
        self.pre_shape_label = ttk.Label(plots, text="", style="Card.TLabel")
        self.pre_shape_label.pack(anchor="w", padx=4, pady=(0, 4))
        self.preprocess_plot = PlotFrame(plots, figsize=(9, 4.5))
        self.preprocess_plot.pack(fill="both", expand=True)

    def _build_preprocessor(self) -> DataPreprocessor:
        drop = []
        if self.drop_address_var.get() and self.df is not None and "Address" in self.df.columns:
            drop.append("Address")
        if self.drop_index_var.get() and self.df is not None and "Unnamed: 0" in self.df.columns:
            drop.append("Unnamed: 0")

        use_pca = self.pca_var.get()
        pca_components = None
        pca_variance = None
        if use_pca:
            raw = self.pca_var_entry.get()
            try:
                val = float(raw)
            except ValueError:
                val = 0.95
            if self.pca_mode_var.get() == "components":
                pca_components = max(2, int(val))
            else:
                pca_variance = min(0.999, max(0.1, float(val)))

        return DataPreprocessor(
            drop_columns=drop,
            numeric_impute_strategy=self.num_impute_var.get(),
            categorical_impute_strategy=self.cat_impute_var.get(),
            scale=self.scale_var.get(),
            use_pca=use_pca,
            pca_components=pca_components,
            pca_variance=pca_variance,
            cap_outliers=self.cap_outliers_var.get(),
        )

    def _run_preprocess(self):
        if self.df is None:
            messagebox.showinfo("No data", "Load a CSV first.")
            return
        self.set_status("Running preprocessing…")
        self.preprocess_status.config(text="Working…")
        self.update_idletasks()

        try:
            self.preprocessor = self._build_preprocessor()
            from preprocessing import split_features_target
            X, y = split_features_target(self.df, TARGET)
            Xt = self.preprocessor.fit_transform(X)
        except Exception as exc:
            messagebox.showerror("Preprocessing error", str(exc))
            self.preprocess_status.config(text=f"Error: {exc}")
            self.set_status("Preprocessing failed.")
            return

        before = X.shape
        after = Xt.shape
        self.pre_shape_label.config(
            text=f"Raw features: {before[1]} columns  →  Transformed: {after[1]} components  "
                 f"(rows kept: {before[0]} → {after[0]})",
            style="Card.TLabel",
        )

        ax = self.preprocess_plot.ax
        ax.clear()
        ev = self.preprocessor.explained_variance_ratio_
        if ev is not None:
            cum = np.cumsum(ev)
            x = np.arange(1, len(ev) + 1)
            ax.bar(x, ev, color=ACCENT, alpha=0.8, label="individual")
            ax.plot(x, cum, color=ACCENT2, marker="o", linewidth=2, label="cumulative")
            ax.axhline(0.95, color=WARN, linestyle="--", linewidth=1)
            ax.set_title("PCA explained variance", color=ACCENT)
            ax.set_xlabel("principal component")
            ax.set_ylabel("variance ratio")
            ax.legend(facecolor=BG2, edgecolor=PANEL)
        else:
            ax.text(0.5, 0.5, "PCA disabled", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        self.preprocess_plot.draw()

        self.preprocess_status.config(text=f"Done — {after[1]} components retained.")
        self.set_status("Preprocessing complete.")

    # --------------------------------------------------------------------- Train
    def _build_train(self):
        root = ScrollFrame(self.tab_train)
        root.pack(fill="both", expand=True)
        body = root.inner

        top = ttk.Frame(body)
        top.pack(fill="x", padx=12, pady=10)

        params = ttk.LabelFrame(top, text="Random Forest hyper-parameters")
        params.pack(side="left", fill="y", padx=(0, 8))

        self.n_estimators_var = tk.IntVar(value=200)
        self.max_depth_var = tk.IntVar(value=0)
        self.min_samples_var = tk.IntVar(value=2)
        self.max_features_var = tk.StringVar(value="sqrt")
        self.test_size_var = tk.DoubleVar(value=0.2)

        def slider_row(parent, label, var, lo, hi, step, fmt="{:.0f}", extra=None):
            ttk.Label(parent, text=label).pack(anchor="w", padx=8, pady=(6, 0))
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=8)
            scale = ttk.Scale(row, from_=lo, to=hi, orient="horizontal",
                              variable=var, length=180)
            scale.pack(side="left")
            val_lbl = ttk.Label(row, text=fmt.format(float(var.get())), width=8)
            val_lbl.pack(side="left", padx=8)
            def update(_=None):
                val_lbl.config(text=fmt.format(float(var.get())))
            scale.bind("<Motion>", update)
            scale.bind("<ButtonRelease-1>", update)
            if extra:
                extra(row, val_lbl, update)
            return update

        slider_row(params, "n_estimators (trees)", self.n_estimators_var, 50, 600, 10)
        ttk.Label(params, text="max_depth (0 = none)").pack(anchor="w", padx=8, pady=(6, 0))
        dp = ttk.Frame(params); dp.pack(fill="x", padx=8)
        ds = ttk.Scale(dp, from_=0, to=40, orient="horizontal", variable=self.max_depth_var, length=180)
        ds.pack(side="left")
        dl = ttk.Label(dp, text="none", width=8); dl.pack(side="left", padx=8)
        def upd_depth(_=None):
            v = int(self.max_depth_var.get())
            dl.config(text="none" if v == 0 else str(v))
        ds.bind("<Motion>", upd_depth); ds.bind("<ButtonRelease-1>", upd_depth)

        ttk.Label(params, text="min_samples_split").pack(anchor="w", padx=8, pady=(6, 0))
        sp = ttk.Frame(params); sp.pack(fill="x", padx=8)
        ss = ttk.Scale(sp, from_=2, to=20, orient="horizontal", variable=self.min_samples_var, length=180)
        ss.pack(side="left")
        sl = ttk.Label(sp, text="2", width=8); sl.pack(side="left", padx=8)
        def upd_min(_=None):
            sl.config(text=str(int(self.min_samples_var.get())))
        ss.bind("<Motion>", upd_min); ss.bind("<ButtonRelease-1>", upd_min)

        ttk.Label(params, text="max_features").pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Combobox(params, textvariable=self.max_features_var, state="readonly",
                     values=["sqrt", "log2", 1.0, 0.5, 0.3], width=10).pack(anchor="w", padx=8, pady=4)

        ttk.Label(params, text=f"Test size ({self.test_size_var.get():.0%})").pack(anchor="w", padx=8, pady=(6, 0))
        tp = ttk.Frame(params); tp.pack(fill="x", padx=8)
        ts = ttk.Scale(tp, from_=0.1, to=0.4, orient="horizontal", variable=self.test_size_var, length=180)
        ts.pack(side="left")
        tl = ttk.Label(tp, text=f"{self.test_size_var.get():.0%}", width=8); tl.pack(side="left", padx=8)
        def upd_test(_=None):
            tl.config(text=f"{self.test_size_var.get():.0%}")
        ts.bind("<Motion>", upd_test); ts.bind("<ButtonRelease-1>", upd_test)

        actions = ttk.Frame(params)
        actions.pack(fill="x", padx=8, pady=10)
        self.train_btn = ttk.Button(actions, text="▶ Train model", style="Accent.TButton",
                                    command=self._start_training)
        self.train_btn.pack(side="left")
        ttk.Button(actions, text="Save model", command=self._save_model).pack(side="left", padx=8)
        ttk.Button(actions, text="Load model", command=self._load_model).pack(side="left")

        self.progress = ttk.Progressbar(params, mode="indeterminate", length=240)
        self.progress.pack(fill="x", padx=8, pady=(4, 8))

        results = ttk.Frame(top)
        results.pack(side="left", fill="both", expand=True, padx=8)
        self.metric_labels: dict[str, ttk.Label] = {}
        grid = ttk.Frame(results, style="Card.TFrame")
        grid.pack(fill="x", pady=(0, 8))
        for i, key in enumerate(["R2 (train)", "R2 (test)", "RMSE (train)", "RMSE (test)", "MAE (train)", "MAE (test)"]):
            card = ttk.Frame(grid, style="Card.TFrame")
            card.grid(row=0, column=i, padx=4, sticky="nsew")
            grid.grid_columnconfigure(i, weight=1)
            ttk.Label(card, text=key, style="Card.TLabel").pack(padx=8, pady=(8, 0))
            lbl = ttk.Label(card, text="—", style="Stat.TLabel")
            lbl.pack(padx=8, pady=(0, 10))
            self.metric_labels[key] = lbl

        self.train_plot = PlotFrame(results, figsize=(8, 4.6))
        self.train_plot.pack(fill="both", expand=True)

        self.after(100, self._poll_train)

    def _start_training(self):
        if self.df is None:
            messagebox.showinfo("No data", "Load a CSV first.")
            return
        if self._train_thread and self._train_thread.is_alive():
            return

        if self.preprocessor is None:
            self.preprocessor = self._build_preprocessor()

        max_features = self.max_features_var.get()
        try:
            max_features = float(max_features)
        except ValueError:
            pass

        max_depth = int(self.max_depth_var.get())
        max_depth = None if max_depth == 0 else max_depth

        self.trainer = ModelTrainer(
            preprocessor=self.preprocessor,
            n_estimators=int(self.n_estimators_var.get()),
            max_depth=max_depth,
            min_samples_split=int(self.min_samples_var.get()),
            max_features=max_features,
        )

        self.train_btn.config(state="disabled")
        self.progress.start(12)
        self.set_status("Training… this may take a moment.")

        self._train_thread = threading.Thread(
            target=self._train_worker,
            args=(self.df, self.test_size_var.get()),
            daemon=True,
        )
        self._train_thread.start()

    def _train_worker(self, df, test_size):
        try:
            result = self.trainer.fit(df, test_size=test_size)
            self.train_queue.put(("ok", result))
        except Exception as exc:
            self.train_queue.put(("err", str(exc)))

    def _poll_train(self):
        try:
            kind, payload = self.train_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_train)
            return

        self.progress.stop()
        self.train_btn.config(state="normal")

        if kind == "err":
            messagebox.showerror("Training error", payload)
            self.set_status("Training failed.")
            self.after(100, self._poll_train)
            return

        result = payload
        for key, value in result.metrics.as_dict().items():
            text = f"{value:.4f}" if "R2" in key else fmt_money(value)
            self.metric_labels[key].config(text=text)

        ax = self.train_plot.ax
        ax.clear()
        if result.importances is not None and result.component_labels:
            top_n = min(15, len(result.importances))
            order = np.argsort(result.importances)[::-1][:top_n]
            vals = result.importances[order]
            labels = [result.component_labels[i] for i in order]
            ax.barh(labels[::-1], vals[::-1], color=ACCENT)
            ax.set_title("Top feature / PCA-component importances", color=ACCENT)
            ax.tick_params(axis="y", labelsize=9)
        self.train_plot.draw()

        r2 = result.metrics.r2_test
        color = SUCCESS if r2 >= 0.7 else (WARN if r2 >= 0.5 else DANGER)
        self.metric_labels["R2 (test)"].config(foreground=color)
        self.set_status(f"Training complete — R² (test) = {r2:.4f}")
        self._maybe_enable_predict()
        self.after(100, self._poll_train)

    def _save_model(self):
        if self.trainer is None or self.trainer.pipeline_ is None:
            messagebox.showinfo("Not trained", "Train a model first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save model", defaultextension=".joblib",
            filetypes=[("Joblib", "*.joblib")],
            initialfile="price_predictor.joblib",
            initialdir=os.path.dirname(MODEL_PATH),
        )
        if not path:
            return
        try:
            self.trainer.save(path)
            self.set_status(f"Model saved → {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def _load_model(self):
        path = filedialog.askopenfilename(
            title="Load model", filetypes=[("Joblib", "*.joblib")],
            initialdir=os.path.dirname(MODEL_PATH),
        )
        if not path:
            return
        try:
            self.trainer = ModelTrainer.load(path)
            self.preprocessor = self.trainer.preprocessor
            if self.trainer.last_result:
                result = self.trainer.last_result
                for key, value in result.metrics.as_dict().items():
                    text = f"{value:.4f}" if "R2" in key else fmt_money(value)
                    self.metric_labels[key].config(text=text)
            self.set_status("Model loaded.")
            self._maybe_enable_predict()
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    # ------------------------------------------------------------------- Predict
    def _build_predict(self):
        root = ScrollFrame(self.tab_predict)
        root.pack(fill="both", expand=True)
        self.predict_body = root.inner

        intro = ttk.Frame(self.predict_body)
        intro.pack(fill="x", padx=12, pady=10)
        ttk.Label(intro, text="Enter property details to estimate its sale price.",
                  style="Muted.TLabel").pack(side="left")
        self.predict_btn = ttk.Button(intro, text="Predict price", style="Accent.TButton",
                                      command=self._do_predict, state="disabled")
        self.predict_btn.pack(side="right")

        self.predict_form = ttk.Frame(self.predict_body)
        self.predict_form.pack(fill="x", padx=12, pady=(0, 8))
        self.predict_fields: dict[str, tk.Widget] = {}

        result_card = self._card(self.predict_body, "Prediction")
        result_card.pack(fill="x", padx=12, pady=6)
        self.predict_result_label = ttk.Label(result_card, text="Train a model to enable predictions.",
                                              style="Big.TLabel")
        self.predict_result_label.pack(padx=12, pady=12)
        self.predict_detail_label = ttk.Label(result_card, text="", style="Muted.TLabel")
        self.predict_detail_label.pack(anchor="w", padx=12, pady=(0, 12))

    def _populate_predict_form(self):
        if self.df is None:
            return
        for child in self.predict_form.winfo_children():
            child.destroy()
        self.predict_fields.clear()

        feature_cols = [c for c in self.df.columns if c != TARGET]
        categorical = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if c != TARGET]

        per_row = 3
        for i, col in enumerate(feature_cols):
            r, c = divmod(i, per_row)
            cell = ttk.Frame(self.predict_form, style="Card.TFrame")
            cell.grid(row=r, column=c, padx=5, pady=5, sticky="ew")
            self.predict_form.grid_columnconfigure(c, weight=1)
            ttk.Label(cell, text=col, style="Card.TLabel").pack(anchor="w", padx=10, pady=(8, 2))

            if col in categorical:
                options = sorted(self.df[col].dropna().astype(str).unique().tolist())
                widget = ttk.Combobox(cell, values=options, width=24)
                if options:
                    widget.set(options[0])
            else:
                series = self.df[col].dropna()
                lo = float(series.min()) if not series.empty else 0.0
                hi = float(series.max()) if not series.empty else 100.0
                default = float(series.median()) if not series.empty else lo
                widget = ttk.Entry(cell, width=26)
                widget.insert(0, f"{default:.2f}")
            widget.pack(anchor="w", padx=10, pady=(0, 8))
            self.predict_fields[col] = widget

    def _collect_record(self) -> dict:
        record = {}
        for col, widget in self.predict_fields.items():
            val = widget.get()
            if isinstance(widget, ttk.Combobox):
                record[col] = val if val else None
            else:
                try:
                    record[col] = float(val)
                except ValueError:
                    record[col] = val
        return record

    def _do_predict(self):
        if self.trainer is None or self.trainer.pipeline_ is None:
            messagebox.showinfo("Not trained", "Train or load a model first.")
            return
        record = self._collect_record()
        try:
            price = self.trainer.predict_one(record)
        except Exception as exc:
            messagebox.showerror("Prediction error", str(exc))
            return

        self.predict_result_label.config(text=fmt_money(price))
        if self.trainer.y_train is not None:
            median = float(self.trainer.y_train.median())
            diff = price - median
            sign = "+" if diff >= 0 else "−"
            self.predict_detail_label.config(
                text=f"City median: {fmt_money(median)}   |   vs median: {sign}{fmt_money(abs(diff))}"
            )
        self.set_status(f"Predicted price: {fmt_money(price)}")

    # ------------------------------------------------------------------ Insights
    def _build_insights(self):
        root = ScrollFrame(self.tab_insights)
        root.pack(fill="both", expand=True)
        body = root.inner

        row1 = ttk.Frame(body)
        row1.pack(fill="both", expand=True, padx=12, pady=10)
        self.insight_corr = PlotFrame(row1, figsize=(6.5, 5))
        self.insight_corr.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.insight_region = PlotFrame(row1, figsize=(6.5, 5))
        self.insight_region.pack(side="left", fill="both", expand=True, padx=(6, 0))

        row2 = ttk.Frame(body)
        row2.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.insight_type = PlotFrame(row2, figsize=(6.5, 4.5))
        self.insight_type.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.insight_scatter = PlotFrame(row2, figsize=(6.5, 4.5))
        self.insight_scatter.pack(side="left", fill="both", expand=True, padx=(6, 0))

    def _refresh_insights(self):
        if self.df is None:
            return
        self._plot_corr()
        self._plot_region()
        self._plot_type()
        self._plot_scatter()

    def _plot_corr(self):
        ax = self.insight_corr.ax
        ax.clear()
        num = self.df.select_dtypes(include=[np.number])
        if TARGET in num.columns and num.shape[1] > 1:
            corr = num.corr(numeric_only=True)
            im = ax.imshow(corr, cmap="magma", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
            ax.set_yticklabels(corr.columns, fontsize=7)
            ax.set_title("Numeric correlation", color=ACCENT, fontsize=10)
            self.insight_corr.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.insight_corr.draw()

    def _plot_region(self):
        ax = self.insight_region.ax
        ax.clear()
        if "Regionname" in self.df.columns and TARGET in self.df.columns:
            order = self.df.groupby("Regionname")[TARGET].median().sort_values().index
            data = [self.df[self.df["Regionname"] == r][TARGET].values for r in order]
            ax.boxplot(data, labels=[r.replace("Metropolitan", "Met").replace("Victoria", "Vic") for r in order],
                       showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=ACCENT, color=ACCENT),
                       medianprops=dict(color=ACCENT2))
            ax.set_title("Price by region", color=ACCENT, fontsize=10)
            ax.tick_params(axis="x", labelsize=7, rotation=25)
            ax.ticklabel_format(style="plain", axis="y")
        self.insight_region.draw()

    def _plot_type(self):
        ax = self.insight_type.ax
        ax.clear()
        if "Type" in self.df.columns and TARGET in self.df.columns:
            groups = self.df.groupby("Type")[TARGET].mean()
            ax.bar(groups.index, groups.values, color=[ACCENT, ACCENT2, SUCCESS])
            ax.set_title("Mean price by property type", color=ACCENT, fontsize=10)
            ax.set_ylabel(TARGET)
            ax.ticklabel_format(style="plain", axis="y")
        self.insight_type.draw()

    def _plot_scatter(self):
        ax = self.insight_scatter.ax
        ax.clear()
        if "BuildingArea" in self.df.columns and TARGET in self.df.columns:
            sub = self.df[(self.df["BuildingArea"] > 0) & (self.df["BuildingArea"] < 1000)]
            ax.scatter(sub["BuildingArea"], sub[TARGET], s=6, alpha=0.4, color=ACCENT)
            ax.set_title("Price vs BuildingArea", color=ACCENT, fontsize=10)
            ax.set_xlabel("BuildingArea")
            ax.set_ylabel(TARGET)
            ax.ticklabel_format(style="plain", axis="y")
        self.insight_scatter.draw()

    def _maybe_enable_predict(self):
        ready = self.trainer is not None and self.trainer.pipeline_ is not None
        self.predict_btn.config(state="normal" if ready else "disabled")



def main():
    app = App()
    def on_tab_change(_):
        app._maybe_enable_predict()
    app.notebook.bind("<<NotebookTabChanged>>", on_tab_change)
    app.mainloop()


if __name__ == "__main__":
    main()