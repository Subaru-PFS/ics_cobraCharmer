#!/usr/bin/env python

import tkinter as tk
from time import sleep
from tkinter import messagebox

from ics.cobraCharmer import ethernet, fpgaLogger
from ics.cobraCharmer.convert import get_per
from ics.cobraCharmer.func import *


class App(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        self.protoLogger = None
        self.spintoggle = 'cw'

        # Reset Button----------------------------------
        frame1 = tk.Frame(self)
        self.hkexport = tk.IntVar()
        btn_cnct = tk.Button(frame1, text="Connect", command=self.connect)
        btn_rst = tk.Button(frame1, text="Reset", command=self.op_reset)
        btn_pwr = tk.Button(frame1, text="Power", command=self.op_power)
        btn_hk = tk.Button(frame1, text="Housekeeping", command=self.op_hk)
        chk_hkexport = tk.Checkbutton(frame1, text="Export Housekeeping", variable=self.hkexport)
        btn_cnct.pack(side="left", fill=None, expand=False)
        btn_rst.pack(side="left", fill=None, expand=False)
        btn_pwr.pack(side="left", fill=None, expand=False)
        btn_hk.pack(side="left", fill=None, expand=False)
        chk_hkexport.pack(side="left", fill=None, expand=False)

        frame1.grid(row=1, sticky=tk.W, padx=4)

        # Board Info-----------------------------------------------
        frame2 = tk.Frame(self)
        lab_brd = tk.Label(frame2, text="Board:")
        lab_cob1 = tk.Label(frame2, text="Cobra Start:")
        lab_cob2 = tk.Label(frame2, text="Cobra end:")
        self.ent_brd = tk.Entry(frame2)
        self.ent_cob1 = tk.Entry(frame2)
        self.ent_cob2 = tk.Entry(frame2)

        lab_brd.pack(side="left", fill=None, expand=False)
        self.ent_brd.pack(side="left", fill=None, expand=False)
        lab_cob1.pack(side="left", fill=None, expand=False)
        self.ent_cob1.pack(side="left", fill=None, expand=False)
        lab_cob2.pack(side="left", fill=None, expand=False)
        self.ent_cob2.pack(side="left", fill=None, expand=False)

        frame2.grid(row=2, sticky=tk.W, padx=4)

        self.ent_brd.insert(10, '1')
        self.ent_cob1.insert(10, '1')
        self.ent_cob2.insert(10, '1')
        self.ent_brd.config(width=5)
        self.ent_cob1.config(width=5)
        self.ent_cob2.config(width=5)

        self.ent_cob1.bind_all("<Up>", func=self.upCobra)
        self.ent_cob1.bind_all("<Down>", func=self.downCobra)

        self.ent_brd.bind_all("<Shift-Up>", func=self.upBoard)
        self.ent_brd.bind_all("<Shift-Down>", func=self.downBoard)
        self.ent_brd.bind_all("<Escape>", self.dropFocus)

        self.ent_brd.bind_all("h", self.help)
        self.ent_brd.bind_all("?", self.help)

        # Motor Info------------------------------------------------
        frame3 = tk.Frame(self)
        self.m1_en = tk.IntVar()
        self.m2_en = tk.IntVar()
        self.m_cw = tk.IntVar()
        chk_m1 = tk.Checkbutton(frame3, text="Theta", variable=self.m1_en)
        chk_m2 = tk.Checkbutton(frame3, text="Phi", variable=self.m2_en)
        chk_m2.select()
        chk_cw = tk.Checkbutton(frame3, text="Clockwise", variable=self.m_cw)
        self.chk_m1 = chk_m1
        self.chk_m2 = chk_m2
        self.chk_cw = chk_cw

        chk_m1.bind_all("t", func=lambda w: chk_m1.toggle())
        chk_m2.bind_all("p", func=lambda w: chk_m2.toggle())
        chk_m2.bind_all("m", self.toggleMotor)
        chk_cw.bind_all("c", func=lambda w: chk_cw.toggle())
        chk_cw.bind_all("d", func=lambda w: chk_cw.toggle())

        chk_m1.pack(side="left", fill=None, expand=False)
        chk_m2.pack(side="left", fill=None, expand=False)
        chk_cw.pack(side="left", fill=None, expand=False)

        frame3.grid(row=3, sticky=tk.W, padx=4)

        # Set Frequency----------------------------------------------
        frame4 = tk.Frame(self)
        btn_set = tk.Button(frame4, text="SetFreq", command=self.op_set)
        lab_setf = tk.Label(frame4, text="Low, High Freq (Khz):")
        self.ent_setf1 = tk.Entry(frame4)
        self.ent_setf2 = tk.Entry(frame4)

        btn_set.pack(side="left", fill=None, expand=False)
        lab_setf.pack(side="left", fill=None, expand=False)
        self.ent_setf1.pack(side="left", fill=None, expand=False)
        self.ent_setf2.pack(side="left", fill=None, expand=False)

        frame4.grid(row=4, sticky=tk.W, padx=4)

        self.ent_setf1.insert(10, "62.8")
        self.ent_setf2.insert(10, "103.2")
        self.ent_setf1.config(width=5)
        self.ent_setf2.config(width=5)

        # Calibrate ------------------------------------------------
        frame5 = tk.Frame(self)
        btn_cal = tk.Button(frame5, text="Cal", command=self.op_cal)
        lab_cal1 = tk.Label(frame5, text="M1 Freq Window (Khz):")
        self.ent_cal1_low = tk.Entry(frame5)
        self.ent_cal1_high = tk.Entry(frame5)
        lab_cal2 = tk.Label(frame5, text="M2 Freq Window (Khz):")
        self.ent_cal2_low = tk.Entry(frame5)
        self.ent_cal2_high = tk.Entry(frame5)

        btn_cal.pack(side="left", fill=None, expand=False)
        lab_cal1.pack(side="left", fill=None, expand=False)
        self.ent_cal1_low.pack(side="left", fill=None, expand=False)
        self.ent_cal1_high.pack(side="left", fill=None, expand=False)
        lab_cal2.pack(side="left", fill=None, expand=False)
        self.ent_cal2_low.pack(side="left", fill=None, expand=False)
        self.ent_cal2_high.pack(side="left", fill=None, expand=False)

        frame5.grid(row=5, sticky=tk.W, padx=4)

        self.ent_cal1_low.insert(10, "55")
        self.ent_cal1_high.insert(10, "70")
        self.ent_cal2_low.insert(10, "95")
        self.ent_cal2_high.insert(10, "120")
        self.ent_cal1_low.config(width=5)
        self.ent_cal1_high.config(width=5)
        self.ent_cal2_low.config(width=5)
        self.ent_cal2_high.config(width=5)

        # Run--------------------------------------------------------
        frame7 = tk.Frame(self)
        btn_run = tk.Button(frame7, text="Run", command=self.op_run)
        btn_run.bind_all("<space>", func=lambda w: btn_run.invoke())
        btn_run.bind_all("<Left>", func=self.run_ccw)
        btn_run.bind_all("<Right>", func=self.run_cw)

        lab_run_time = tk.Label(frame7, text="Time(mSec):")
        lab_run_steps = tk.Label(frame7, text="Steps:")
        self.ent_run_time = tk.Entry(frame7)
        self.ent_run_steps = tk.Entry(frame7)

        btn_run.pack(side="left", fill=None, expand=False)
        lab_run_time.pack(side="left", fill=None, expand=False)
        self.ent_run_time.pack(side="left", fill=None, expand=False)
        lab_run_steps.pack(side="left", fill=None, expand=False)
        self.ent_run_steps.pack(side="left", fill=None, expand=False)

        frame7.grid(row=7, sticky=tk.W, padx=4)

        self.ent_run_time.insert(10, "0.07")
        self.ent_run_steps.insert(10, "50")
        self.ent_run_time.config(width=5)
        self.ent_run_steps.config(width=5)

        # Message
        self.msg_str = tk.StringVar()
        msg = tk.Message(self, textvariable=self.msg_str)
        msg.config(aspect=300, width=500)
        msg.grid(row=15, sticky=tk.W, padx=4)

        # Log Text Display
        self.txtLog = tk.Text(self, height=35, width=90, bg='#d0d0d0')
        self.txtLog.config(state=tk.DISABLED)
        self.txtLog.grid(row=20, column=0, sticky=tk.W)
        scroll = tk.Scrollbar(self)
        scroll.grid(row=20, column=2, sticky=tk.NS)
        scroll.config(command=self.txtLog.yview)
        self.txtLog.config(yscrollcommand=scroll.set)

    def dropFocus(self, event):
        self.focus()

    def update(self):
        # Update log text in GUI
        if self.protoLogger is None:
            return

        path = self.protoLogger.logger.handlers[0].baseFilename
        with open(path) as f:
            logData = f.read()

        self.txtLog.config(state=tk.NORMAL)
        self.txtLog.delete(1.0, tk.END)
        self.txtLog.insert(tk.END, logData)
        self.txtLog.see(tk.END)
        self.txtLog.config(state=tk.DISABLED)

    def help(self, event):
        """Dump help strings to text field. """

        helpStr = """
    ====== Selection:
    t, p           toggle theta, phi motor control
    c, d           change motor direction.
    <Up>,<Down>    select higher or lower cobra(s).
                      Always kept within the current board.
    <Shift-Up>, <Shift-Down>
                   select the next or previous board.
                      Upper cobra might be clipped from 29 to 28.

    ====== Motion:
    <Left>,<Right> run the selected cobras CCW or CW
    <Space>        run the selected cobra.

    The ontime and #steps are taken from the GUI fields.

    ====== Misc:
    Hit ESC to get out of text fields.

    board:        1..84
    cobra limits: 1..29 for odd boards, 1..28 for even boards

    Housekeeping always shows entire single board.
    Calibration limits differ from original GUI.

    """

        self.txtLog.config(state=tk.NORMAL)
        self.txtLog.delete(1.0, tk.END)
        self.txtLog.insert(tk.END, helpStr)
        self.txtLog.see(tk.END)
        self.txtLog.config(state=tk.DISABLED)

    def connect(self):
        self.protoLogger = fpgaLogger.FPGAProtocolLogger(logRoot='.')
        ethernet.sock.connect('fpga', 4001, protoLogger=self.protoLogger)
        self.update()
        self.msg_str.set("Connected.")
        self.update()

    def op_reset(self):
        er = RST()
        if not er:
            sleep(0.8)
            DIA()
        txt = "Reset Error!" if er else "Reset Succeeded."
        self.msg_str.set(txt)
        self.update()

    def op_power(self):
        er = POW()
        if not er:
            sleep(0.8)
            DIA()
        txt = "Power Error!" if er else "Power Succeeded."
        self.msg_str.set(txt)
        self.update()

    def genCobra(self, board, cobraNum):
        module = (board - 1)//2 + 1
        cobraInModule = (cobraNum - 1)*2 + 1
        if board%2 == 0:
            cobraInModule += 1
        return Cobra(module, cobraInModule)

    def op_hk(self):
        board = int(self.ent_brd.get())
        cobras = [self.genCobra(board, i) for i in range(1, 30)]
        for c in cobras:
            c.p = HkParams(m0=(0, 1000), m1=(0, 1000))
        er = HK(cobras, self.hkexport.get())
        txt = "Hk Error!" if er else "Hk Ran Successfully."
        self.msg_str.set(txt)
        self.update()

    def op_cal(self):
        board = int(self.ent_brd.get())
        c_low = int(self.ent_cob1.get())
        c_high = int(self.ent_cob2.get())
        cobras = [self.genCobra(board, i) for i in range(c_low, c_high+1)]
        m0_low = get_per(float(self.ent_cal1_high.get()))
        m0_high = get_per(float(self.ent_cal1_low.get()))
        m1_low = get_per(float(self.ent_cal2_high.get()))
        m1_high = get_per(float(self.ent_cal2_low.get()))
        m0rng = (m0_low, m0_high)
        m1rng = (m1_low, m1_high)
        spin = CW_DIR if self.m_cw.get() else CCW_DIR
        en = (self.m1_en.get(), self.m2_en.get())
        for c in cobras:
            c.p = CalParams(m0=m0rng, m1=m1rng, en=en, dir=spin)
        self.msg_str.set("Calibrate running...")
        er = CAL(cobras)
        txt = "Cal Error!" if er else "Cal Ran Successfully."
        self.msg_str.set(txt)
        self.update()

    def op_set(self):
        board = int(self.ent_brd.get())
        c_low = int(self.ent_cob1.get())
        c_high = int(self.ent_cob2.get())
        cobras = [self.genCobra(board, i) for i in range(c_low, c_high+1)]
        p0 = get_per(float(self.ent_setf1.get()))
        p1 = get_per(float(self.ent_setf2.get()))
        en = (self.m1_en.get(), self.m2_en.get())
        for c in cobras:
            c.p = SetParams(p0=p0, p1=p1, en=en)
        er = SET(cobras)
        txt = "SetFreq Error!" if er else "SetFreq Ran Successfully."
        self.msg_str.set(txt)
        self.update()

    def op_run(self):
        board = int(self.ent_brd.get())
        c_low = int(self.ent_cob1.get())
        c_high = int(self.ent_cob2.get())
        cobras = [self.genCobra(board, i) for i in range(c_low, c_high+1)]
        rtime = int(1000 * float(self.ent_run_time.get()))
        if rtime < 15 or rtime > 140:
            messagebox.showerror("Error", "Please enter a time between 0.015 and 0.140 mSec")
            return
        steps = int(self.ent_run_steps.get())
        spin = CW_DIR if self.m_cw.get() else CCW_DIR
        en = (self.m1_en.get(), self.m2_en.get())
        for c in cobras:
            c.p = RunParams(pu=(rtime, rtime), st=(steps, steps), en=en, dir=spin)
        self.msg_str.set("Running...")
        er = RUN(cobras)
        txt = "Run Error!" if er else "Run Ran Successfully."
        self.msg_str.set(txt)
        self.update()

    def run_dir(self, event, direction):
        board = int(self.ent_brd.get())
        c_low = int(self.ent_cob1.get())
        c_high = int(self.ent_cob2.get())
        cobras = [self.genCobra(board, i) for i in range(c_low, c_high+1)]
        rtime = int(1000 * float(self.ent_run_time.get()))
        if rtime < 15 or rtime > 140:
            messagebox.showerror("Error", "Please enter a time between 0.015 and 0.140 mSec")
            return
        steps = int(self.ent_run_steps.get())
        spin = direction
        en = (self.m1_en.get(), self.m2_en.get())
        for c in cobras:
            c.p = RunParams(pu=(rtime, rtime), st=(steps, steps), en=en, dir=spin)
        self.msg_str.set("Running...")
        er = RUN(cobras)
        txt = "Run Error!" if er else "Run Ran Successfully."
        self.msg_str.set(txt)
        self.update()

    def run_ccw(self, event):
        self.run_dir(event, CCW_DIR)

    def run_cw(self, event):
        self.run_dir(event, CW_DIR)

    def upCobra(self, event):
        board = int(self.ent_brd.get())
        c_low = int(self.ent_cob1.get())
        c_high = int(self.ent_cob2.get())

        if board % 2 == 1:
            if c_high >= 29:
                return
        elif c_high >= 28:
            return

        self.ent_cob1.delete(0, tk.END)
        self.ent_cob2.delete(0, tk.END)
        self.ent_cob1.insert(10, str(c_low+1))
        self.ent_cob2.insert(10, str(c_high+1))

    def downCobra(self, event):
        board = int(self.ent_brd.get())
        c_low = int(self.ent_cob1.get())
        c_high = int(self.ent_cob2.get())

        if c_low <= 1:
            return
        self.ent_cob1.delete(0, tk.END)
        self.ent_cob2.delete(0, tk.END)
        self.ent_cob1.insert(10, str(c_low-1))
        self.ent_cob2.insert(10, str(c_high-1))

    def clipCobraNum(self):
        board = int(self.ent_brd.get())
        c_low = int(self.ent_cob1.get())
        c_high = int(self.ent_cob2.get())

        cobraLim = 29 if (board % 2 == 1) else 28
        if c_high > cobraLim:
            self.ent_cob2.delete(0, tk.END)
            self.ent_cob2.insert(10, str(cobraLim))
        if c_low > cobraLim:
            self.ent_cob1.delete(0, tk.END)
            self.ent_cob1.insert(10, str(cobraLim))

    def upBoard(self, event):
        board = int(self.ent_brd.get())

        if board >= 84:
            return
        self.ent_brd.delete(0, tk.END)
        self.ent_brd.insert(10, str(board+1))

        self.clipCobraNum()

    def downBoard(self, event):
        board = int(self.ent_brd.get())
        if board < 2:
            return
        self.ent_brd.delete(0, tk.END)
        self.ent_brd.insert(10, str(board-1))

        self.clipCobraNum()

    def toggleMotor(self, event):
        self.chk_m1.toggle()
        self.chk_m2.toggle()


root = tk.Tk()


# assign closing routine
def on_close():
    ethernet.sock.close()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)

app = App(root)
app.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
root.wm_title("Driver Board GUI")

root.mainloop()
