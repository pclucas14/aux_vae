from collections import OrderedDict as OD

DEBUGGER = OD()

def get_debugger():
    global DEBUGGER
    return DEBUGGER

def get_debug_values():
    global DEBUGGER
    return list(DEBUGGER.values())

def get_debug_keys():
    global DEBUGGER
    return list(DEBUGGER.keys())

def print_keys():
    i = 0
    for k in DEBUGGER.keys():
        print( '%s : %s' % (i, k))
        i += 1
