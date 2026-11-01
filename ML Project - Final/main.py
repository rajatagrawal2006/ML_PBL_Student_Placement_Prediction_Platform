# Main Tkinter GUI application for Student Performance Predictor

import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd

import data_utils
import models


def busy(method):
    """Decorator setting cursor to 'watch' and handling unexpected errors in a dialog."""
    def wrapper(self, *args, **kwargs):
        try:
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            return method(self, *args, **kwargs)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            try:
                self.root.config(cursor="")
            except Exception:
                pass
    return wrapper


def fmt_num(val):
    """Formats whole numbers without decimals and floats to 2 decimal places."""
    if val is None:
        return "--"
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        return f"{f:.2f}"
    except (ValueError, TypeError):
        return str(val)


class StudentPerformanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Performance Predictor")
        self.root.geometry("1120x740")
        self.root.minsize(1000, 680)

        # State storage
        self.dataset_name = "None"
        self.X, self.y_class, self.y_reg = None, None, None
        self.class_names = ("Fail", "Pass")
        self.split_cache = None
        self.predict_entries = {}

        self.setup_styles()
        self.build_ui()
        self.load_initial_data()

    def setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.root.configure(bg="#f3f5f9")
        self.style.configure(".", background="#f3f5f9", foreground="#22314d", font=("Segoe UI", 9))
        self.style.configure("TFrame", background="#f3f5f9")
        self.style.configure("TNotebook", background="#f3f5f9", borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#e2e8f0", foreground="#22314d", padding=(12, 6), font=("Segoe UI", 9, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", "#2f6fed")])
        self.style.configure("TButton", background="#ffffff", foreground="#22314d", padding=(8, 4), borderwidth=1)
        self.style.configure("Accent.TButton", background="#2f6fed", foreground="#ffffff", padding=(10, 5), font=("Segoe UI", 9, "bold"))
        self.style.map("Accent.TButton", background=[("active", "#1f55d6"), ("pressed", "#1f55d6")])
        self.style.configure("Treeview", background="#ffffff", foreground="#22314d", rowheight=24, fieldbackground="#ffffff")
        self.style.configure("Treeview.Heading", background="#eef2f8", foreground="#22314d", font=("Segoe UI", 9, "bold"))
        self.style.map("Treeview", background=[("selected", "#d0e1fd")], foreground=[("selected", "#1f55d6")])

    def make_card(self, parent, title, val="--", val_color="#22314d"):
        box = tk.Frame(parent, bg="#ffffff", highlightbackground="#e2e8f0", highlightthickness=1, padx=12, pady=7)
        lbl_t = tk.Label(box, text=title, font=("Segoe UI", 8), fg="#6b7790", bg="#ffffff")
        lbl_t.pack(anchor="w")
        lbl_v = tk.Label(box, text=val, font=("Segoe UI", 15, "bold"), fg=val_color, bg="#ffffff")
        lbl_v.pack(anchor="w", pady=(2, 0))
        return box, lbl_t, lbl_v

    def make_plot_canvas(self, parent):
        fig = Figure(figsize=(5.0, 3.6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        w = canvas.get_tk_widget()
        w.config(width=10, height=10)
        w.pack(fill="both", expand=True)
        return fig, canvas

    def make_treeview(self, parent, cols, headings=None, height=8):
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=height)
        headings = headings or cols
        for c, h in zip(cols, headings):
            tree.heading(c, text=h)
            tree.column(c, width=90, anchor="center")
        sy = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sy.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        return tree

    def build_ui(self):
        # Header
        h_frame = tk.Frame(self.root, bg="#ffffff", padx=16, pady=8, relief="solid", borderwidth=1)
        h_frame.pack(fill=tk.X)
        tk.Label(h_frame, text="Student Performance Predictor", font=("Segoe UI", 15, "bold"), fg="#22314d", bg="#ffffff").pack(anchor="w")
        tk.Label(h_frame, text="Second-Year B.Tech AI Project: Classification, Regression, Clustering & Inference", font=("Segoe UI", 8), fg="#6b7790", bg="#ffffff").pack(anchor="w")

        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.tab_data, self.tab_clf, self.tab_reg, self.tab_cluster, self.tab_predict = [ttk.Frame(self.notebook) for _ in range(5)]
        for t, name in zip(
            [self.tab_data, self.tab_clf, self.tab_reg, self.tab_cluster, self.tab_predict],
            ["1. Data", "2. Classification", "3. Regression", "4. Clustering", "5. Predict"]
        ):
            self.notebook.add(t, text=name)

        # Status Bar
        self.status_bar = tk.Frame(self.root, bg="#ffffff", padx=12, pady=4, relief="solid", borderwidth=1)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.lbl_status = tk.Label(self.status_bar, text="No dataset loaded", font=("Segoe UI", 8), fg="#6b7790", bg="#ffffff")
        self.lbl_status.pack(side=tk.LEFT)

        self.init_data_tab()
        self.init_classification_tab()
        self.init_regression_tab()
        self.init_clustering_tab()
        self.init_predict_tab()

    # 1. Data Tab
    def init_data_tab(self):
        lp = ttk.LabelFrame(self.tab_data, text="Choose a dataset", padding=10)
        lp.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=6)
        ttk.Button(lp, text="Student performance data", style="Accent.TButton", command=self.on_load_student).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(lp, text="Load my own CSV", command=self.on_load_csv).pack(fill=tk.X, pady=(0, 8))
        ttk.Separator(lp).pack(fill=tk.X, pady=6)
        ttk.Label(lp, text="Or make synthetic data", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(2, 4))
        
        rf = ttk.Frame(lp); rf.pack(fill=tk.X, pady=2)
        ttk.Label(rf, text="Rows:").pack(side=tk.LEFT)
        self.spn_synth_rows = ttk.Spinbox(rf, from_=100, to=1000, increment=50, width=6)
        self.spn_synth_rows.set(395); self.spn_synth_rows.pack(side=tk.RIGHT)

        nf = ttk.Frame(lp); nf.pack(fill=tk.X, pady=2)
        ttk.Label(nf, text="Noise:").pack(side=tk.LEFT)
        self.sld_noise = ttk.Scale(nf, from_=0.0, to=1.0, value=0.25, length=85)
        self.sld_noise.pack(side=tk.RIGHT)
        ttk.Button(lp, text="Generate data", command=self.on_generate_synth).pack(fill=tk.X, pady=(6, 0))

        rp = ttk.Frame(self.tab_data); rp.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=6)
        cf = ttk.Frame(rp); cf.pack(fill=tk.X, pady=(0, 6))
        c1, _, self.lbl_card_rows = self.make_card(cf, "Total Rows"); c1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        c2, _, self.lbl_card_feats = self.make_card(cf, "Total Factors"); c2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        c3, self.lbl_card_c0_t, self.lbl_card_c0 = self.make_card(cf, "Fail (class 0)", val_color="#c5221f"); c3.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        c4, self.lbl_card_c1_t, self.lbl_card_c1 = self.make_card(cf, "Pass (class 1)", val_color="#1e8e3e"); c4.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        self.lbl_data_info = tk.Label(rp, text="Dataset: None | Numeric regression target: None", font=("Segoe UI", 9, "bold"), fg="#22314d", bg="#f3f5f9")
        self.lbl_data_info.pack(anchor="w", padx=4, pady=(0, 4))
        p_box = ttk.LabelFrame(rp, text="Preview (first 5 rows, first 8 factors)", padding=6)
        p_box.pack(fill=tk.BOTH, expand=True)
        self.tree_preview = self.make_treeview(p_box, ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "Result"])

    # 2. Classification Tab
    def init_classification_tab(self):
        ctrl = ttk.LabelFrame(self.tab_clf, text="Model Controls", padding=8); ctrl.pack(fill=tk.X, padx=4, pady=5)
        ttk.Label(ctrl, text="Model:").pack(side=tk.LEFT, padx=3)
        self.cmb_clf_m = ttk.Combobox(ctrl, values=models.CLASSIFIERS, state="readonly", width=18); self.cmb_clf_m.set(models.CLASSIFIERS[0]); self.cmb_clf_m.pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text="Scaling:").pack(side=tk.LEFT, padx=3)
        self.cmb_clf_s = ttk.Combobox(ctrl, values=models.SCALERS, state="readonly", width=14); self.cmb_clf_s.set(models.SCALERS[0]); self.cmb_clf_s.pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text="Features:").pack(side=tk.LEFT, padx=3)
        self.cmb_clf_f = ttk.Combobox(ctrl, values=models.FEATURE_METHODS, state="readonly", width=12); self.cmb_clf_f.set(models.FEATURE_METHODS[0]); self.cmb_clf_f.pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Train and evaluate", style="Accent.TButton", command=self.on_train_clf).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Compare all models", command=self.on_compare_clf).pack(side=tk.LEFT, padx=3)

        cf = ttk.Frame(self.tab_clf); cf.pack(fill=tk.X, padx=4, pady=(0, 5))
        _, _, self.lbl_clf_acc = self.make_card(cf, "Accuracy"); _.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        _, _, self.lbl_clf_prec = self.make_card(cf, "Precision"); _.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        _, _, self.lbl_clf_rec = self.make_card(cf, "Recall"); _.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        _, _, self.lbl_clf_f1 = self.make_card(cf, "F1 Score"); _.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        body = ttk.Frame(self.tab_clf); body.pack(fill=tk.BOTH, expand=True, padx=4, pady=3)
        self.lf_clf_t = ttk.LabelFrame(body, text="Evaluation Output", padding=6); self.lf_clf_t.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))
        self.tree_clf = self.make_treeview(self.lf_clf_t, ["#", "Actual", "Predicted", "Outcome"])
        pf = ttk.LabelFrame(body, text="Confusion Matrix", padding=6); pf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(3, 0))
        self.fig_clf, self.canvas_clf = self.make_plot_canvas(pf)

    # 3. Regression Tab
    def init_regression_tab(self):
        ctrl = ttk.LabelFrame(self.tab_reg, text="Regression Controls", padding=8); ctrl.pack(fill=tk.X, padx=4, pady=5)
        ttk.Label(ctrl, text="Method:").pack(side=tk.LEFT, padx=3)
        self.cmb_reg_m = ttk.Combobox(ctrl, values=models.REGRESSORS, state="readonly", width=16); self.cmb_reg_m.set(models.REGRESSORS[0]); self.cmb_reg_m.pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text="Learning Rate:").pack(side=tk.LEFT, padx=3)
        self.spn_reg_lr = ttk.Spinbox(ctrl, from_=0.01, to=0.5, increment=0.01, width=6); self.spn_reg_lr.set(0.1); self.spn_reg_lr.pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text="Max Iter:").pack(side=tk.LEFT, padx=3)
        self.spn_reg_iter = ttk.Spinbox(ctrl, from_=100, to=5000, increment=100, width=6); self.spn_reg_iter.set(2000); self.spn_reg_iter.pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Train and evaluate", style="Accent.TButton", command=self.on_train_reg).pack(side=tk.LEFT, padx=6)

        cf = ttk.Frame(self.tab_reg); cf.pack(fill=tk.X, padx=4, pady=(0, 5))
        _, _, self.lbl_reg_mae = self.make_card(cf, "MAE"); _.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        _, _, self.lbl_reg_mse = self.make_card(cf, "MSE"); _.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        _, _, self.lbl_reg_rmse = self.make_card(cf, "RMSE"); _.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        _, _, self.lbl_reg_r2 = self.make_card(cf, "R² Score"); _.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        _, _, self.lbl_reg_adj = self.make_card(cf, "Adjusted R²"); _.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        pf = ttk.LabelFrame(self.tab_reg, text="Regression Plots", padding=6); pf.pack(fill=tk.BOTH, expand=True, padx=4, pady=3)
        self.fig_reg, self.canvas_reg = self.make_plot_canvas(pf)

    # 4. Clustering Tab
    def init_clustering_tab(self):
        ctrl = ttk.LabelFrame(self.tab_cluster, text="K-Means Configuration", padding=8); ctrl.pack(fill=tk.X, padx=4, pady=5)
        ttk.Label(ctrl, text="Clusters (k):").pack(side=tk.LEFT, padx=3)
        self.spn_cluster_k = ttk.Spinbox(ctrl, from_=2, to=6, width=5); self.spn_cluster_k.set(3); self.spn_cluster_k.pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Run K-means", style="Accent.TButton", command=self.on_run_kmeans).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Show elbow plot", command=self.on_show_elbow).pack(side=tk.LEFT, padx=3)

        cf = ttk.Frame(self.tab_cluster); cf.pack(fill=tk.X, padx=4, pady=(0, 5))
        _, _, self.lbl_clust_k = self.make_card(cf, "Clusters (k)"); _.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        _, _, self.lbl_clust_inertia = self.make_card(cf, "Inertia (WCSS)"); _.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        body = ttk.Frame(self.tab_cluster); body.pack(fill=tk.BOTH, expand=True, padx=4, pady=3)
        self.lf_clust_t = ttk.LabelFrame(body, text="Cluster Summary", padding=6); self.lf_clust_t.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))
        self.tree_cluster = self.make_treeview(self.lf_clust_t, ["Cluster", "Students", "Percentage"])
        pf = ttk.LabelFrame(body, text="Visualization", padding=6); pf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(3, 0))
        self.fig_cluster, self.canvas_cluster = self.make_plot_canvas(pf)

    # 5. Predict Tab
    def init_predict_tab(self):
        top = ttk.LabelFrame(self.tab_predict, text="Inference Pipeline Configuration", padding=8); top.pack(fill=tk.X, padx=4, pady=5)
        ttk.Label(top, text="Model:").pack(side=tk.LEFT, padx=3)
        self.cmb_pred_m = ttk.Combobox(top, values=models.CLASSIFIERS, state="readonly", width=18); self.cmb_pred_m.set("Logistic Regression"); self.cmb_pred_m.pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="Scaling:").pack(side=tk.LEFT, padx=3)
        self.cmb_pred_s = ttk.Combobox(top, values=models.SCALERS, state="readonly", width=14); self.cmb_pred_s.set("StandardScaler"); self.cmb_pred_s.pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="Features:").pack(side=tk.LEFT, padx=3)
        self.cmb_pred_f = ttk.Combobox(top, values=models.FEATURE_METHODS, state="readonly", width=12); self.cmb_pred_f.set("SelectKBest"); self.cmb_pred_f.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Predict student performance", style="Accent.TButton", command=self.on_predict_sample).pack(side=tk.LEFT, padx=8)

        body = ttk.Frame(self.tab_predict); body.pack(fill=tk.BOTH, expand=True, padx=4, pady=3)
        self.lf_factors = ttk.LabelFrame(body, text="Student Factors (Top 8 Predictive Factors)", padding=8)
        self.lf_factors.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.form_container = ttk.Frame(self.lf_factors); self.form_container.pack(fill=tk.BOTH, expand=True)

        res_p = ttk.LabelFrame(body, text="Prediction Result", padding=10); res_p.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        c_res = tk.Frame(res_p, bg="#ffffff", highlightbackground="#e2e8f0", highlightthickness=1, padx=16, pady=12); c_res.pack(fill=tk.X, pady=(0, 8))
        tk.Label(c_res, text="PREDICTED OUTCOME", font=("Segoe UI", 9, "bold"), fg="#6b7790", bg="#ffffff").pack(anchor="w")
        self.lbl_pred_outcome = tk.Label(c_res, text="Awaiting Prediction", font=("Segoe UI", 20, "bold"), fg="#22314d", bg="#ffffff")
        self.lbl_pred_outcome.pack(anchor="w", pady=(4, 0))

        c_prob = tk.Frame(res_p, bg="#ffffff", highlightbackground="#e2e8f0", highlightthickness=1, padx=16, pady=10); c_prob.pack(fill=tk.X, pady=(0, 8))
        tk.Label(c_prob, text="CONFIDENCE PROBABILITY", font=("Segoe UI", 9, "bold"), fg="#6b7790", bg="#ffffff").pack(anchor="w")
        self.lbl_pred_prob = tk.Label(c_prob, text="--", font=("Segoe UI", 12, "bold"), fg="#22314d", bg="#ffffff")
        self.lbl_pred_prob.pack(anchor="w", pady=(3, 0))

        lf_in = ttk.LabelFrame(res_p, text="Input Factor Values", padding=6); lf_in.pack(fill=tk.BOTH, expand=True)
        self.tree_pred_in = self.make_treeview(lf_in, ["Factor", "Entered Value"])

    def load_initial_data(self):
        try:
            self.on_load_student()
        except Exception:
            pass

    @busy
    def on_load_student(self):
        X, yc, yr, cnames = data_utils.load_student_dataset()
        self.set_dataset("student-mat.csv (Student Performance)", X, yc, yr, cnames)

    @busy
    def on_load_csv(self):
        path = filedialog.askopenfilename(title="Select CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        col = simpledialog.askstring("Target Column", "Enter the target column name:")
        if not col:
            return
        X, yc, yr, cnames = data_utils.load_csv_dataset(path, col.strip())
        self.set_dataset(f"CSV ({os.path.basename(path)})", X, yc, yr, cnames)

    @busy
    def on_generate_synth(self):
        n = int(self.spn_synth_rows.get())
        noise = float(self.sld_noise.get())
        X, yc, yr, cnames = data_utils.generate_synthetic_data(n, noise)
        self.set_dataset(f"Synthetic Data (Noise={noise:.2f})", X, yc, yr, cnames)

    def set_dataset(self, name, X, y_class, y_reg, class_names):
        self.dataset_name, self.X, self.y_class, self.y_reg, self.class_names = name, X, y_class, y_reg, class_names
        self.split_cache = models.split_data(self.X, self.y_class, self.y_reg)
        X_tr, X_te, _, _, _, _ = self.split_cache

        self.lbl_card_rows.config(text=fmt_num(len(X)))
        self.lbl_card_feats.config(text=fmt_num(X.shape[1]))
        self.lbl_card_c0_t.config(text=f"{class_names[0]} (class 0)")
        self.lbl_card_c0.config(text=fmt_num((y_class == 0).sum()))
        self.lbl_card_c1_t.config(text=f"{class_names[1]} (class 1)")
        self.lbl_card_c1.config(text=fmt_num((y_class == 1).sum()))

        reg_info = f"Available ({fmt_num(y_reg.min())} - {fmt_num(y_reg.max())})" if y_reg is not None else "None"
        self.lbl_data_info.config(text=f"Dataset: {name} | Numeric regression target: {reg_info}")
        self.lbl_status.config(text=f"Dataset: {name} | Rows: {len(X)} | Factors: {X.shape[1]} | Train: {len(X_tr)}, Test: {len(X_te)} (80/20 split)")

        # Preview table (first 8 factors, Result, Numeric target)
        cols = list(X.columns[:8]) + ["Result"]
        if y_reg is not None:
            cols.append("Numeric target")
        self.tree_preview.delete(*self.tree_preview.get_children())
        self.tree_preview["columns"] = cols
        for c in cols:
            self.tree_preview.heading(c, text=c)
            self.tree_preview.column(c, width=95, anchor="center")
        
        head_df, head_yc = X.head(5), y_class.head(5).to_numpy()
        head_yr = y_reg.head(5).to_numpy() if y_reg is not None else None
        for i in range(len(head_df)):
            row = [fmt_num(head_df.iloc[i][c]) for c in X.columns[:8]]
            row.append(class_names[1] if head_yc[i] == 1 else class_names[0])
            if head_yr is not None:
                row.append(fmt_num(head_yr[i]))
            self.tree_preview.insert("", "end", values=row)

        self.refresh_predict_inputs()

    def ensure_data(self):
        if self.X is None or self.split_cache is None:
            messagebox.showwarning("No Data", "Please choose or generate a dataset in the Data tab first.")
            return False
        return True

    @busy
    def on_train_clf(self):
        if not self.ensure_data():
            return
        m, s, f = self.cmb_clf_m.get(), self.cmb_clf_s.get(), self.cmb_clf_f.get()
        X_tr, X_te, y_tr, y_te, _, _ = self.split_cache
        res = models.train_classifier(m, s, f, X_tr, y_tr, X_te, y_te)
        met, y_pred, cm = res["metrics"], res["y_pred"], res["cm"]

        self.lbl_clf_acc.config(text=fmt_num(met["Accuracy"]))
        self.lbl_clf_prec.config(text=fmt_num(met["Precision"]))
        self.lbl_clf_rec.config(text=fmt_num(met["Recall"]))
        self.lbl_clf_f1.config(text=fmt_num(met["F1 Score"]))

        self.lf_clf_t.config(text=f"Sample Test Predictions: {m}")
        cols = ["#", "Actual", "Predicted", "Outcome"]
        self.tree_clf.delete(*self.tree_clf.get_children())
        self.tree_clf["columns"] = cols
        for c in cols:
            self.tree_clf.heading(c, text=c)
            self.tree_clf.column(c, width=75, anchor="center")
        y_te_arr = y_te.to_numpy()
        for i in range(min(12, len(y_te))):
            act = self.class_names[1] if y_te_arr[i] == 1 else self.class_names[0]
            pred = self.class_names[1] if y_pred[i] == 1 else self.class_names[0]
            self.tree_clf.insert("", "end", values=[i + 1, act, pred, "Correct" if act == pred else "Wrong"])

        self.fig_clf.clear()
        ax = self.fig_clf.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        self.fig_clf.colorbar(im, ax=ax)
        ax.set_title(f"Confusion Matrix: {m}", fontsize=10, pad=8)
        ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels([self.class_names[0], self.class_names[1]], fontsize=8)
        ax.set_yticklabels([self.class_names[0], self.class_names[1]], fontsize=8)
        for i in range(2):
            for j in range(2):
                col = "white" if cm[i, j] > (cm.max() / 2.0) else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=col, fontweight="bold")
        self.fig_clf.tight_layout()
        self.canvas_clf.draw()

    @busy
    def on_compare_clf(self):
        if not self.ensure_data():
            return
        X_tr, X_te, y_tr, y_te, _, _ = self.split_cache
        res = models.compare_all_models(X_tr, y_tr, X_te, y_te, self.cmb_clf_s.get(), self.cmb_clf_f.get())
        self.lf_clf_t.config(text="Model Comparison (All 5 Classifiers)")
        cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
        self.tree_clf.delete(*self.tree_clf.get_children())
        self.tree_clf["columns"] = cols
        for c in cols:
            self.tree_clf.heading(c, text=c)
            self.tree_clf.column(c, width=90, anchor="center")
        for r in res:
            self.tree_clf.insert("", "end", values=[r["Model"], fmt_num(r["Accuracy"]), fmt_num(r["Precision"]), fmt_num(r["Recall"]), fmt_num(r["F1 Score"])])

    @busy
    def on_train_reg(self):
        if not self.ensure_data():
            return
        if self.y_reg is None:
            messagebox.showinfo("No Target", "Regression requires a continuous numeric target (y_reg is None).\nPlease choose student or synthetic data.")
            for lbl in [self.lbl_reg_mae, self.lbl_reg_mse, self.lbl_reg_rmse, self.lbl_reg_r2, self.lbl_reg_adj]:
                lbl.config(text="--")
            self.fig_reg.clear(); self.canvas_reg.draw(); return

        m = self.cmb_reg_m.get()
        lr, max_i = float(self.spn_reg_lr.get()), int(self.spn_reg_iter.get())
        X_tr, X_te, _, _, yr_tr, yr_te = self.split_cache
        res = models.train_regression(m, X_tr, yr_tr, X_te, yr_te, lr=lr, max_iter=max_i)
        met, y_pred, hist = res["metrics"], res["y_pred"], res["history"]

        self.lbl_reg_mae.config(text=fmt_num(met["MAE"]))
        self.lbl_reg_mse.config(text=fmt_num(met["MSE"]))
        self.lbl_reg_rmse.config(text=fmt_num(met["RMSE"]))
        self.lbl_reg_r2.config(text=fmt_num(met["R2"]))
        self.lbl_reg_adj.config(text=fmt_num(met["Adjusted R2"]))

        self.fig_reg.clear()
        y_te_arr = yr_te.to_numpy()
        if m == "Gradient Descent":
            ax1 = self.fig_reg.add_subplot(121)
            ax1.scatter(y_te_arr, y_pred, color="#2f6fed", alpha=0.7, edgecolors="k", s=25)
            mn, mx = min(float(y_te_arr.min()), float(y_pred.min())), max(float(y_te_arr.max()), float(y_pred.max()))
            ax1.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="y=x")
            ax1.set_title("Actual vs Predicted", fontsize=9); ax1.legend(fontsize=7); ax1.grid(True, linestyle=":", alpha=0.5)
            ax2 = self.fig_reg.add_subplot(122)
            ax2.plot(hist, color="#c5221f", linewidth=1.5)
            ax2.set_title("Cost Convergence Curve", fontsize=9); ax2.grid(True, linestyle=":", alpha=0.5)
        else:
            ax = self.fig_reg.add_subplot(111)
            ax.scatter(y_te_arr, y_pred, color="#2f6fed", alpha=0.7, edgecolors="k", s=30)
            mn, mx = min(float(y_te_arr.min()), float(y_pred.min())), max(float(y_te_arr.max()), float(y_pred.max()))
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="y=x")
            ax.set_title(f"{m} Regression", fontsize=10); ax.legend(fontsize=8); ax.grid(True, linestyle=":", alpha=0.5)
        self.fig_reg.tight_layout(); self.canvas_reg.draw()

    @busy
    def on_run_kmeans(self):
        if not self.ensure_data():
            return
        k = int(self.spn_cluster_k.get())
        labels, pca_2d, counts, inertia = models.run_kmeans_clustering(self.X, k)
        self.lbl_clust_k.config(text=fmt_num(k)); self.lbl_clust_inertia.config(text=fmt_num(inertia))
        self.lf_clust_t.config(text=f"Cluster Sizes (k={k})")
        cols = ["Cluster", "Students", "Percentage"]
        self.tree_cluster.delete(*self.tree_cluster.get_children())
        self.tree_cluster["columns"] = cols
        for c in cols:
            self.tree_cluster.heading(c, text=c); self.tree_cluster.column(c, width=80, anchor="center")
        tot = len(self.X)
        for cid, count in sorted(counts.items()):
            self.tree_cluster.insert("", "end", values=[f"Cluster {cid}", fmt_num(count), f"{(count/tot)*100:.1f}%"])

        self.fig_cluster.clear()
        ax = self.fig_cluster.add_subplot(111)
        ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=labels, cmap="tab10", alpha=0.8, edgecolors="k", s=30)
        ax.set_title(f"K-Means (k={k}) via 2D PCA", fontsize=10); ax.grid(True, linestyle=":", alpha=0.5)
        self.fig_cluster.tight_layout(); self.canvas_cluster.draw()

    @busy
    def on_show_elbow(self):
        if not self.ensure_data():
            return
        k_vals, inertias = models.calculate_elbow_curve(self.X, max_k=8)
        self.lf_clust_t.config(text="Elbow Curve Values")
        cols = ["k", "Inertia (WCSS)"]
        self.tree_cluster.delete(*self.tree_cluster.get_children())
        self.tree_cluster["columns"] = cols
        for c in cols:
            self.tree_cluster.heading(c, text=c); self.tree_cluster.column(c, width=100, anchor="center")
        for kv, iv in zip(k_vals, inertias):
            self.tree_cluster.insert("", "end", values=[kv, fmt_num(iv)])

        self.fig_cluster.clear()
        ax = self.fig_cluster.add_subplot(111)
        ax.plot(k_vals, inertias, marker="o", color="#2f6fed", linewidth=2, markersize=5)
        ax.set_title("Elbow Method for Optimal k", fontsize=10); ax.set_xticks(k_vals); ax.grid(True, linestyle="--", alpha=0.6)
        self.fig_cluster.tight_layout(); self.canvas_cluster.draw()

    def refresh_predict_inputs(self):
        for w in self.form_container.winfo_children():
            w.destroy()
        self.predict_entries.clear()
        if self.X is None or self.split_cache is None:
            return
        X_tr, _, y_tr, _, _, _ = self.split_cache
        top_cols = models.top_features(X_tr, y_tr, n=8)
        for col in top_cols:
            rf = ttk.Frame(self.form_container); rf.pack(fill=tk.X, pady=3)
            ttk.Label(rf, text=col, width=22, anchor="w").pack(side=tk.LEFT)
            s = self.X[col]
            med_str = fmt_num(float(s.median()))
            u = list(s.unique())
            if len(u) == 2 and set(u).issubset({0, 1, 0.0, 1.0}):
                ent = ttk.Combobox(rf, values=["0", "1"], width=10, state="readonly"); ent.set(med_str)
            else:
                ent = ttk.Entry(rf, width=12); ent.insert(0, med_str)
            ent.pack(side=tk.RIGHT, padx=4)
            self.predict_entries[col] = ent

    @busy
    def on_predict_sample(self):
        if not self.ensure_data():
            return
        X_tr, _, y_tr, _, _, _ = self.split_cache
        sample_dict = X_tr.median().to_dict()
        rows = []
        for col, ent in self.predict_entries.items():
            val = float(ent.get().strip())
            sample_dict[col] = val
            rows.append((col, fmt_num(val)))

        sample_df = pd.DataFrame([sample_dict])[list(self.X.columns)]
        pipe = models.fit_classifier_for_prediction(self.cmb_pred_m.get(), self.cmb_pred_s.get(), self.cmb_pred_f.get(), X_tr, y_tr)
        pred = int(pipe.predict(sample_df)[0])

        res_str = self.class_names[1] if pred == 1 else self.class_names[0]
        self.lbl_pred_outcome.config(text=f"Result: {res_str}", fg="#1e8e3e" if pred == 1 else "#c5221f")

        if hasattr(pipe, "predict_proba"):
            p0, p1 = pipe.predict_proba(sample_df)[0] * 100.0
            self.lbl_pred_prob.config(text=f"{self.class_names[1]}: {p1:.1f}%  |  {self.class_names[0]}: {p0:.1f}%")
        else:
            self.lbl_pred_prob.config(text="Deterministic classification")

        self.tree_pred_in.delete(*self.tree_pred_in.get_children())
        for r in rows:
            self.tree_pred_in.insert("", "end", values=r)


def main():
    root = tk.Tk()
    app = StudentPerformanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
