#!/usr/bin/env python3

import fileinput
import re

rgx = re.compile(r"^(.*landing C(\d+) from.*ctrl=1, idx=(\w\w\w\w), data=(\w\w\w\w).*)$")
for line in fileinput.input("sim.log", inplace=True):
    if (m := rgx.match(line)) and m.groups()[1] not in ['22','23']:
        ctrl = m.groups()[2] + m.groups()[3]
        bctrl = bin(int(ctrl, 16))[2:].zfill(32)
        cmd0 = bctrl[31-6:31-6+3] + bctrl[31-24:31-24+3]
        cmd1 = bctrl[31-9:31-9+3] + bctrl[31-27:31-27+3]
        cmd2 = bctrl[31-12:31-12+3] + bctrl[31-0] + bctrl[31-29:31-29+2]
        cmd3 = bctrl[31-15:31-15+3] + bctrl[31-3:31-3+3]
        c0 = "ADV" if cmd0[-2:] == "01" else "RST" if cmd0[-2:] == "10" else "NOP"
        c1 = "ADV" if cmd1[-2:] == "01" else "RST" if cmd1[-2:] == "10" else "NOP"
        c2 = "ADV" if cmd2[-2:] == "01" else "RST" if cmd2[-2:] == "10" else "NOP"
        c3 = "ADV" if cmd3[-2:] == "01" else "RST" if cmd3[-2:] == "10" else "NOP"
        ttl0 = int(cmd0[:4], 2)
        ttl1 = int(cmd1[:4], 2)
        ttl2 = int(cmd2[:4], 2)
        ttl3 = int(cmd3[:4], 2)
        cmds = [(c0,ttl0), (c1,ttl1), (c2,ttl2), (c3,ttl3)]
        if cmds[-1] == ("NOP", 0):
            cmds = cmds[:3]
            if cmds[-1] == ("NOP", 0):
                cmds = cmds[:2]
                if cmds[-1] == ("NOP", 0):
                    cmds = cmds[:1]
        # cmd_str = " ".join(f"{c}" if ttl == 0 else f"{c}({ttl})" for (c, ttl) in cmds)
        cmd_str = " ".join(f"{c}({ttl})" for (c, ttl) in cmds)
        print(f"{m.groups()[0]} # {cmd_str}")
    else:
        print(line, end='')

