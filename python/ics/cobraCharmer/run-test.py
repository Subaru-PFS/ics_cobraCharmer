from ast import literal_eval as make_tuple
from time import sleep

from ics.cobraCharmer.log import short_log, LOGS
from ics.cobraCharmer.ethernet import sock
import ics.cobraCharmer.main as main
from ics.cobraCharmer.convert import get_per
from ics.cobraCharmer.func import *


def update():
    pass


def connect():
    main.setup()
    # self.update()
    print("Connected.")


def op_set():
    board = int(1)
    c_low = int(62.8)
    c_high = int(103.2)
    cobras = [Cobra(board, i) for i in range(c_low, c_high+1)]
    p0 = get_per(float(self.ent_setf1.get()))
    p1 = get_per(float(self.ent_setf2.get()))
    en = (self.m1_en.get(), self.m2_en.get())
    for c in cobras:
        c.p = SetParams(p0=p0, p1=p1, en=en)
    er = SET(cobras)
    txt = "SetFreq Error!" if er else "SetFreq Ran Successfully."


connect()
op_set()
