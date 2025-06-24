import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import numpy as np
import matplotlib.font_manager as fm


def get_available_chinese_fonts():
    """获取系统中可用的中文字体"""
    chinese_fonts = []
    for font in fm.findSystemFonts():
        try:
            font_prop = fm.FontProperties(fname=font)
            if font_prop.get_name() and (
                    'hei' in font_prop.get_name().lower() or
                    'song' in font_prop.get_name().lower() or
                    'kai' in font_prop.get_name().lower() or
                    'fang' in font_prop.get_name().lower()
            ):
                chinese_fonts.append(font_prop.get_name())
        except:
            continue
    return list(set(chinese_fonts))  # 去重


# 获取可用的中文字体
available_fonts = get_available_chinese_fonts()

# 设置中文字体
plt.rcParams["font.family"] = available_fonts if available_fonts else ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class WeatherAnalysisSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("气象数据分析与可视化系统")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        self.data = None
        self.file_path = None

        # 创建菜单栏
        self.create_menu()

        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 文件信息
        file_info_frame = ttk.Frame(control_frame)
        file_info_frame.pack(fill=tk.X, pady=5)
        ttk.Label(file_info_frame, text="当前文件:").pack(anchor=tk.W)
        self.file_label = ttk.Label(file_info_frame, text="未选择文件", foreground="red")
        self.file_label.pack(anchor=tk.W)

        # 数据信息
        data_info_frame = ttk.LabelFrame(control_frame, text="数据信息", padding="5")
        data_info_frame.pack(fill=tk.X, pady=5)

        self.rows_var = tk.StringVar(value="行数: --")
        self.columns_var = tk.StringVar(value="列数: --")
        self.stats_var = tk.StringVar(value="统计数据: 未计算")

        ttk.Label(data_info_frame, textvariable=self.rows_var).pack(anchor=tk.W)
        ttk.Label(data_info_frame, textvariable=self.columns_var).pack(anchor=tk.W)
        ttk.Label(data_info_frame, textvariable=self.stats_var).pack(anchor=tk.W)

        # 分析选项
        analysis_frame = ttk.LabelFrame(control_frame, text="分析选项", padding="5")
        analysis_frame.pack(fill=tk.X, pady=5)

        # 数据列选择
        ttk.Label(analysis_frame, text="选择数据列:").pack(anchor=tk.W, pady=2)
        self.column_combo = ttk.Combobox(analysis_frame, state="disabled")
        self.column_combo.pack(fill=tk.X, pady=2)

        # 图表类型选择
        ttk.Label(analysis_frame, text="选择图表类型:").pack(anchor=tk.W, pady=2)
        self.chart_types = [
            "折线图", "柱状图", "箱线图", "散点图",
            "直方图", "热力图", "相关性矩阵"
        ]
        self.chart_combo = ttk.Combobox(analysis_frame, values=self.chart_types)
        self.chart_combo.current(0)
        self.chart_combo.pack(fill=tk.X, pady=2)

        # 分析按钮
        self.analyze_btn = ttk.Button(
            analysis_frame, text="执行分析",
            command=self.perform_analysis,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(fill=tk.X, pady=10)

        # 统计分析按钮
        self.stats_btn = ttk.Button(
            analysis_frame, text="统计分析",
            command=self.show_statistics,
            state=tk.DISABLED
        )
        self.stats_btn.pack(fill=tk.X, pady=5)

        # 右侧显示区域
        display_frame = ttk.LabelFrame(main_frame, text="数据可视化", padding="10")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建图表区域
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开文件", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit, accelerator="Ctrl+Q")
        menubar.add_cascade(label="文件", menu=file_menu)

        # 分析菜单
        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="执行分析", command=self.perform_analysis, accelerator="F5")
        analysis_menu.add_command(label="统计分析", command=self.show_statistics)
        menubar.add_cascade(label="分析", menu=analysis_menu)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        help_menu.add_command(label="使用帮助", command=self.show_help)
        menubar.add_cascade(label="帮助", menu=help_menu)

        self.root.config(menu=menubar)

        # 绑定快捷键
        self.root.bind("<Control-o>", lambda event: self.open_file())
        self.root.bind("<Control-q>", lambda event: self.root.quit())
        self.root.bind("<F5>", lambda event: self.perform_analysis() if self.data is not None else None)

    def open_file(self, event=None):
        """打开气象数据文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("CSV 文件", "*.csv"),
                ("Excel 文件", "*.xlsx *.xls"),
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ],
            title="选择气象数据文件"
        )

        if file_path:
            self.file_path = file_path
            self.file_label.config(text=os.path.basename(file_path), foreground="black")
            self.status_var.set(f"正在加载文件: {os.path.basename(file_path)}")
            self.root.update()

            try:
                # 根据文件扩展名选择读取方法
                if file_path.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    self.data = pd.read_excel(file_path)
                elif file_path.endswith('.txt'):
                    # 尝试自动检测分隔符
                    with open(file_path, 'r') as f:
                        first_line = f.readline()
                        if '\t' in first_line:
                            self.data = pd.read_csv(file_path, sep='\t')
                        elif ';' in first_line:
                            self.data = pd.read_csv(file_path, sep=';')
                        else:
                            self.data = pd.read_csv(file_path, sep=',')

                # 更新数据信息
                self.rows_var.set(f"行数: {len(self.data)}")
                self.columns_var.set(f"列数: {len(self.data.columns)}")
                self.stats_var.set(f"统计数据: 已加载 {len(self.data.columns)} 列数据")

                # 更新列选择下拉框
                self.column_combo['values'] = list(self.data.columns)
                self.column_combo.current(0)
                self.column_combo['state'] = 'readonly'

                # 启用分析按钮
                self.analyze_btn['state'] = tk.NORMAL
                self.stats_btn['state'] = tk.NORMAL

                self.status_var.set(f"文件加载成功: {os.path.basename(file_path)}")
                messagebox.showinfo("成功", f"文件 '{os.path.basename(file_path)}' 加载成功！")

                # 自动执行初步分析
                self.perform_analysis()

            except Exception as e:
                self.status_var.set("文件加载失败")
                messagebox.showerror("错误", f"加载文件时出错: {str(e)}")
                self.data = None
                self.file_label.config(text="未选择文件", foreground="red")

    def perform_analysis(self):
        """执行数据分析和可视化"""
        if self.data is None:
            messagebox.showwarning("警告", "请先加载数据文件！")
            return

        try:
            self.status_var.set("正在分析数据...")
            self.root.update()

            # 清除当前图表
            self.ax.clear()

            # 获取用户选择
            selected_column = self.column_combo.get()
            chart_type = self.chart_combo.get()

            # 根据选择的图表类型生成相应的可视化
            if chart_type == "折线图":
                self.plot_line_chart(selected_column)
            elif chart_type == "柱状图":
                self.plot_bar_chart(selected_column)
            elif chart_type == "箱线图":
                self.plot_boxplot(selected_column)
            elif chart_type == "散点图":
                self.plot_scatter(selected_column)
            elif chart_type == "直方图":
                self.plot_histogram(selected_column)
            elif chart_type == "热力图":
                self.plot_heatmap()
            elif chart_type == "相关性矩阵":
                self.plot_correlation_matrix()

            # 更新画布
            self.figure.tight_layout()
            self.canvas.draw()

            self.status_var.set(f"数据分析完成: {chart_type}")

        except Exception as e:
            self.status_var.set("分析过程中出错")
            messagebox.showerror("错误", f"分析数据时出错: {str(e)}")

    def plot_line_chart(self, column):
        """绘制折线图"""
        # 尝试将索引转换为日期时间（如果适用）
        try:
            if pd.api.types.is_numeric_dtype(self.data.index):
                self.ax.plot(self.data.index, self.data[column])
            else:
                # 尝试将第一列作为x轴（假设是日期）
                first_column = self.data.columns[0]
                self.data[first_column] = pd.to_datetime(self.data[first_column])
                self.ax.plot(self.data[first_column], self.data[column])
                self.ax.tick_params(axis='x', rotation=45)
        except:
            self.ax.plot(self.data[column])

        self.ax.set_title(f"{column} 趋势分析")
        self.ax.set_ylabel(column)
        self.ax.grid(True, linestyle='--', alpha=0.7)

    def plot_bar_chart(self, column):
        """绘制柱状图"""
        if pd.api.types.is_numeric_dtype(self.data[column]):
            # 对于数值数据，计算平均值并按类别分组（如果有分类列）
            if len(self.data.columns) > 1:
                # 尝试找到合适的分类列
                for col in self.data.columns:
                    if col != column and pd.api.types.is_object_dtype(self.data[col]):
                        # 计算每个类别的平均值
                        grouped_data = self.data.groupby(col)[column].mean()
                        self.ax.bar(grouped_data.index, grouped_data.values)
                        self.ax.set_title(f"{column} 按 {col} 分组的平均值")
                        self.ax.set_xlabel(col)
                        self.ax.tick_params(axis='x', rotation=45)
                        return
            # 如果没有找到合适的分类列，使用前10个数据点
            self.ax.bar(range(10), self.data[column].head(10))
            self.ax.set_title(f"{column} 前10个数据点")
            self.ax.set_xlabel("数据点索引")
        else:
            # 对于分类数据，计算每个类别的数量
            value_counts = self.data[column].value_counts()
            self.ax.bar(value_counts.index, value_counts.values)
            self.ax.set_title(f"{column} 类别分布")
            self.ax.set_xlabel(column)
            self.ax.tick_params(axis='x', rotation=45)

        self.ax.set_ylabel("数量")
        self.ax.grid(True, linestyle='--', alpha=0.7)

    def plot_boxplot(self, column):
        """绘制箱线图"""
        if pd.api.types.is_numeric_dtype(self.data[column]):
            self.ax.boxplot(self.data[column].dropna())
            self.ax.set_title(f"{column} 箱线图分析")
            self.ax.set_ylabel(column)
            self.ax.set_xticklabels([column])
            self.ax.grid(True, linestyle='--', alpha=0.7)
        else:
            messagebox.showwarning("警告", f"列 '{column}' 不是数值类型，无法绘制箱线图！")

    def plot_scatter(self, column):
        """绘制散点图"""
        if pd.api.types.is_numeric_dtype(self.data[column]):
            # 尝试找到另一个数值列作为y轴
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns.remove(column)

            if numeric_columns:
                y_column = numeric_columns[0]
                self.ax.scatter(self.data[column], self.data[y_column])
                self.ax.set_title(f"{column} vs {y_column} 散点图")
                self.ax.set_xlabel(column)
                self.ax.set_ylabel(y_column)
                self.ax.grid(True, linestyle='--', alpha=0.7)
            else:
                messagebox.showwarning("警告", "数据中没有足够的数值列来绘制散点图！")
        else:
            messagebox.showwarning("警告", f"列 '{column}' 不是数值类型，无法绘制散点图！")

    def plot_histogram(self, column):
        """绘制直方图"""
        if pd.api.types.is_numeric_dtype(self.data[column]):
            self.ax.hist(self.data[column].dropna(), bins=20, alpha=0.7, edgecolor='black')
            self.ax.set_title(f"{column} 分布直方图")
            self.ax.set_xlabel(column)
            self.ax.set_ylabel("频率")
            self.ax.grid(True, linestyle='--', alpha=0.7)
        else:
            messagebox.showwarning("警告", f"列 '{column}' 不是数值类型，无法绘制直方图！")

    def plot_heatmap(self):
        """绘制热力图"""
        # 选择数值列
        numeric_data = self.data.select_dtypes(include=[np.number])

        if len(numeric_data.columns) > 1:
            # 计算相关系数矩阵
            corr_matrix = numeric_data.corr()

            # 绘制热力图
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=self.ax)
            self.ax.set_title("气象数据热力图")
        else:
            messagebox.showwarning("警告", "数据中没有足够的数值列来绘制热力图！")

    def plot_correlation_matrix(self):
        """绘制相关性矩阵图"""
        # 选择数值列
        numeric_data = self.data.select_dtypes(include=[np.number])

        if len(numeric_data.columns) > 1:
            # 计算相关系数矩阵
            corr_matrix = numeric_data.corr()

            # 创建相关性矩阵图
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                        cmap='viridis', square=True, ax=self.ax)
            self.ax.set_title("气象数据相关性矩阵")
        else:
            messagebox.showwarning("警告", "数据中没有足够的数值列来绘制相关性矩阵！")

    def show_statistics(self):
        """显示统计分析结果"""
        if self.data is None:
            messagebox.showwarning("警告", "请先加载数据文件！")
            return

        # 创建统计分析窗口
        stats_window = tk.Toplevel(self.root)
        stats_window.title("统计分析结果")
        stats_window.geometry("800x600")

        # 创建文本框显示统计信息
        text_widget = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)

        # 添加滚动条
        scrollbar = tk.Scrollbar(text_widget, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)

        # 计算并显示基本统计信息
        stats_text = "数据基本统计信息:\n\n"
        stats_text += self.data.describe().to_string()

        # 添加缺失值统计
        missing_values = self.data.isnull().sum()
        stats_text += "\n\n缺失值统计:\n"
        stats_text += missing_values[missing_values > 0].to_string()

        if missing_values.sum() == 0:
            stats_text += "无缺失值"

        # 添加数据类型信息
        stats_text += "\n\n数据类型信息:\n"
        stats_text += self.data.dtypes.to_string()

        # 在文本框中显示统计信息
        text_widget.insert(tk.END, stats_text)
        text_widget.config(state=tk.DISABLED)  # 设置为只读

    def show_about(self):
        """显示关于信息"""
        messagebox.showinfo(
            "关于",
            "气象数据分析与可视化系统 v1.0\n\n"
            "这是一个基于Python的气象数据分析工具，"
            "可以帮助用户读取、分析和可视化气象数据。\n\n"
            "支持的文件格式: CSV, Excel, 文本文件\n"
            "支持的图表类型: 折线图、柱状图、箱线图、散点图、直方图、热力图、相关性矩阵"
        )

    def show_help(self):
        """显示使用帮助"""
        help_text = (
            "气象数据分析与可视化系统使用帮助\n\n"
            "1. 打开文件:\n"
            "   - 点击菜单栏中的'文件'->'打开文件'"
            "   - 选择要分析的气象数据文件(支持CSV, Excel, 文本文件)\n\n"
            "2. 数据分析:\n"
            "   - 选择要分析的数据列"
            "   - 选择要生成的图表类型"
            "   - 点击'执行分析'按钮生成图表\n\n"
            "3. 统计分析:\n"
            "   - 点击'统计分析'按钮查看数据的基本统计信息\n\n"
            "4. 快捷键:\n"
            "   - Ctrl+O: 打开文件\n"
            "   - Ctrl+Q: 退出程序\n"
            "   - F5: 执行分析"
        )

        help_window = tk.Toplevel(self.root)
        help_window.title("使用帮助")
        help_window.geometry("600x400")

        # 创建文本框显示帮助信息
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)

        # 添加滚动条
        scrollbar = tk.Scrollbar(text_widget, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)

        # 在文本框中显示帮助信息
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)  # 设置为只读


def main():
    root = tk.Tk()
    app = WeatherAnalysisSystem(root)
    root.mainloop()


if __name__ == "__main__":
    main()