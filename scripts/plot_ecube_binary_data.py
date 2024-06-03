# flake8: noqa
import os
import sys
from sys import platform
from tkinter import filedialog, messagebox
import tkinter as tk
import neuraltoolkit as ntk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class NeuralToolkitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Toolkit Data Loader")

        # Get screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set window size to 80% of the screen size
        window_width = int(screen_width * 0.4)
        window_height = int(screen_height * 0.8)

        self.root.geometry(f"{window_width}x{window_height}+0+0")
        # self.root.geometry("800x1000+0+0")
        self.root.resizable(True, True)
        # self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.file_path = tk.StringVar()
        self.num_channels_var = tk.IntVar(value=64)
        self.hstype_entry = tk.StringVar(value='APT_PCB')
        # self.hstype_entry = tk.StringVar(
        #     value="""APT_PCB,APT_PCB,APT_PCB,APT_PCB,APT_PCB,
        #     APT_PCB,APT_PCB,APT_PCB"""
        # )
        self.ts_scale = tk.IntVar(value=0)
        self.samples_scale = tk.IntVar(value=25000)
        self.nprobes_var = tk.IntVar(value=1)
        self.probenum_var = tk.IntVar(value=0)
        self.probechans_var = tk.IntVar(value=64)
        self.ntet_var = tk.IntVar(value=0)
        self.filter_var = tk.BooleanVar()
        self.sampling_rate_var = tk.IntVar(value=25000)

        self.canvas = None  # Placeholder for the canvas

        self.create_widgets()

    def create_widgets(self):
        def complete_path(event):
            typed = self.file_path.get()
            print(f'typed {typed}')
            if os.path.isdir(typed):
                suggestions = os.listdir(typed)
                print(f'i suggestions {suggestions}')
                print(f'i suggestions[0] {suggestions[0]}')
                if suggestions:
                    self.file_path.set(os.path.join(typed, suggestions[0]))
            else:
                dir_path, partial = os.path.split(typed)
                print(f'dir_path {dir_path} partial {partial}')
                if os.path.isdir(dir_path):
                    suggestions = [f for f in os.listdir(dir_path) if f.startswith(partial)]
                    if suggestions:
                        print(f'i suggestions {suggestions}')
                        print(f'e suggestions[0] {suggestions[0]}')
                        self.file_path.set(os.path.join(dir_path, suggestions[0]))
            return 'break'

        tk.Label(self.root, text="Select Raw File").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        file_path_entry = tk.Entry(self.root, textvariable=self.file_path, width=64)
        file_path_entry.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        file_path_entry.bind("<Tab>", complete_path)
        tk.Button(self.root, text="Browse", command=self.select_file).grid(row=0, column=2, padx=5, pady=2)
        # tk.Label(self.root, text="Select Raw File").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        # tk.Entry(self.root, textvariable=self.file_path, width=64).grid(row=0, column=1, padx=5, pady=2, sticky='w')
        # tk.Button(self.root, text="Browse", command=self.select_file).grid(row=0, column=2, padx=5, pady=2)

        tk.Label(self.root, text="HSType (comma separated)").grid(row=1, column=0, padx=5, pady=2, sticky='e')
        tk.Entry(self.root, textvariable=self.hstype_entry, width=64).grid(row=1, column=1, padx=5, pady=2, sticky='w')

        tk.Label(self.root, text="TS").grid(row=2, column=0, padx=5, pady=2, sticky='e')
        self.ts_scale_widget = tk.Scale(self.root, from_=0, to=25000 * 5 * 60,
                                        resolution=2500,
                                        orient=tk.HORIZONTAL,
                                        length=450,
                                        variable=self.ts_scale)
        self.ts_scale_widget.grid(row=2, column=1, padx=5, pady=2, sticky='w')
        self.ts_scale_widget.bind("<ButtonPress-1>", self.update_scale_limit)

        tk.Label(self.root, text="Samples (25 to 2_500_000)").grid(row=3, column=0, padx=5, pady=2, sticky='e')
        self.samples_scale_widget = tk.Scale(self.root, from_=25, to=25000 * 100,
                                             resolution=25,
                                             orient=tk.HORIZONTAL,
                                             length=450,
                                             variable=self.samples_scale)
        self.samples_scale_widget.grid(row=3, column=1, padx=5, pady=2, sticky='w')

        tk.Label(self.root, text="Sampling Rate").grid(row=4, column=0, padx=5, pady=2, sticky='e')
        tk.Entry(self.root, textvariable=self.sampling_rate_var, width=7).grid(row=4, column=1, padx=5, pady=2, sticky='w')

        tk.Label(self.root, text="Number of Channels").grid(row=5, column=0, padx=5, pady=2, sticky='e')
        tk.Spinbox(self.root, from_=64, to=640, increment=64, textvariable=self.num_channels_var, width=6).grid(row=5, column=1, padx=5, pady=2, sticky='w')


        tk.Label(self.root, text="NProbes").grid(row=6, column=0, padx=5, pady=2, sticky='e')
        tk.Spinbox(self.root, from_=1, to=10, textvariable=self.nprobes_var, width=6).grid(row=6, column=1, padx=5, pady=2, sticky='w')

        tk.Label(self.root, text="ProbeNum").grid(row=7, column=0, padx=5, pady=2, sticky='e')
        tk.Spinbox(self.root, from_=0, to=10, textvariable=self.probenum_var, width=6).grid(row=7, column=1, padx=5, pady=2, sticky='w')

        tk.Label(self.root, text="ProbeChans").grid(row=8, column=0, padx=5, pady=2, sticky='e')
        tk.Spinbox(self.root, from_=64, to=64, textvariable=self.probechans_var, width=6).grid(row=8, column=1, padx=5, pady=2, sticky='w')

        tk.Label(self.root, text="NTet").grid(row=9, column=0, padx=5, pady=2, sticky='e')
        tk.Spinbox(self.root, from_=0, to=15, textvariable=self.ntet_var, width=6).grid(row=9, column=1, padx=5, pady=2, sticky='w')

        tk.Checkbutton(self.root, text="Apply Bandpass Filter", variable=self.filter_var).grid(row=10, column=0, columnspan=3, padx=5, pady=2)

        tk.Button(self.root, text="Load and Plot Data[NTet, TS:TS+Samples]", command=self.load_data).grid(row=11, column=0, columnspan=3, padx=5, pady=10)

    def select_file(self):
        # file = filedialog.askopenfilename(initialdir='/hlabhome/kiranbn/git/neuraltoolkit/scripts/', title="Select file")
        if platform == "darwin":
            default_folder = '/Volumes/'
        elif platform == 'linux':
            default_folder = '/media/'
        else:
            default_folder = ''
        file = filedialog.askopenfilename(initialdir=default_folder, title="Select file")
        self.file_path.set(file)

    def update_scale_limit(self, event):
        rawfile = self.file_path.get()
        number_of_channels = int(self.num_channels_var.get())
        limit = ntk.find_samples_per_chan(rawfile, number_of_channels, 1)
        self.ts_scale_widget.config(to=limit)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()
            plt.close()
            sys.exit()

    def load_data(self):
        try:
            rawfile = self.file_path.get()
            number_of_channels = int(self.num_channels_var.get())
            hstype = self.hstype_entry.get().split(',')
            ts = int(self.ts_scale.get())
            samples = int(self.samples_scale.get())
            te = ts + samples
            nprobes = int(self.nprobes_var.get())
            probenum = int(self.probenum_var.get())
            probechans = int(self.probechans_var.get())
            ntet = int(self.ntet_var.get())
            sampling_rate = int(self.sampling_rate_var.get())

            t, dgc = ntk.load_raw_gain_chmap_1probe(
                rawfile, number_of_channels, hstype, nprobes=nprobes,
                lraw=1, ts=ts, te=te, probenum=probenum, probechans=probechans
            )

            if self.filter_var.get():
                data = ntk.butter_bandpass(dgc, 500, 7500, sampling_rate, 3)
            else:
                data = dgc

            self.plot_data(data, ntet)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_data(self, data, ntet):
        try:
            if self.canvas:
                self.canvas.get_tk_widget().grid_forget()  # Remove existing canvas

            plt.close('all')
            fig, ax = plt.subplots(nrows=4, ncols=1)
            color_list = ['#008080', '#ff7f50', '#a0db8e', '#b0e0e6']
            offset = 0
            tmp_channels = ntk.get_tetrode_channels_from_tetnum(ntet, ch_grp_size=4)
            ts = int(self.ts_scale.get())
            samples = int(self.samples_scale.get())
            for i, channel in enumerate(tmp_channels):
                if i == 3:
                    # Generate xtick positions
                    xticks = np.arange(ts, ts+samples, 1)
                    ax[i].plot(xticks, data[channel, :] + i * offset,
                               color=color_list[i],
                               label=f'Ch{channel} NTet {ntet}')
                    ax[i].legend(loc='upper right')
                    del xticks
                    ax[i].set_xlabel('Samples, ' +
                                     '[Samples/' +
                                     str(self.sampling_rate_var.get()) +
                                     ' = Seconds]')
                    ax[i].set_ylabel('Amplitude [\u03bcV]')
                    ax[i].spines['top'].set_visible(False)
                    ax[i].spines['right'].set_visible(False)
                    ax[i].spines['bottom'].set_visible(False)
                    ax[i].spines['left'].set_visible(False)
                    ax[i].ticklabel_format(style='plain')
                    # Adjust 'pad' for padding
                    ax[i].tick_params(axis='x', which='major', pad=50)

                else:
                    ax[i].plot(data[channel, :] + i * offset,
                               color=color_list[i],
                               label=f'Ch{channel}')
                    ax[i].legend(loc='upper right')
                    ax[i].set_xlabel('')
                    ax[i].set_ylabel('')
                    ax[i].spines['top'].set_visible(False)
                    ax[i].spines['right'].set_visible(False)
                    # X AXIS -BORDER
                    ax[i].spines['bottom'].set_visible(False)
                    ax[i].set_xticklabels([])
                    ax[i].set_xticks([])
                    ax[i].axes.get_xaxis().set_visible(False)
                    # Y AXIS -BORDER
                    ax[i].spines['left'].set_visible(False)
                    # ax[i].set_yticklabels([])
                    # ax[i].set_yticks([])
                    # ax[i].axes.get_yaxis().set_visible(False)

                # if i == 0:
                #   ax[i].set_title(f'Plot for ntet = {ntet}')
                # Adjust overall layout
                fig.tight_layout()
                # Adjust vertical spacing between subplots
                # fig.subplots_adjust(hspace=0.5)

            self.canvas = FigureCanvasTkAgg(fig, master=self.root)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=13, columnspan=3, padx=5, pady=1)
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralToolkitApp(root)
    root.mainloop()
